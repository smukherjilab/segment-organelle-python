import numpy as np
import scipy.ndimage as ndi
from pathlib import Path
from nd2reader import ND2Reader
from skimage import util,io,filters,measure,morphology

def get_nd2_size(path:str):
    with ND2Reader(path) as images:
        size = images.sizes
    return size

def load_nd2_plane(path:str,frame:str='cyx',axes:str='tz',idx:int=0):
    """read an image flexibly with ND2Reader."""
    with ND2Reader(path) as images:
        images.bundle_axes = frame
        images.iter_axes = axes
        img = images[idx]
    return img.squeeze()

def find_rings_otsu_zbyz(img_objt,img_mask):
    """
    img_objt: 3d grayscale image of vacuoles
    img_mask: 2d label image of cells
    """
    img_out = np.zeros_like(img_objt,dtype=bool)
    properties_cell = measure.regionprops(img_mask)
    for pcell in properties_cell:
        min_row,min_col,max_row,max_col = pcell.bbox
        img_cube = img_objt[:,min_row:max_row,min_col:max_col]
        for z in range(img_cube.shape[0]):
            img_cube[z] = img_cube[z]*pcell.image
            th_otsu  = filters.threshold_otsu(img_cube[z])
            img_cbbn = (img_cube[z] > th_otsu)
            img_out[z,min_row:max_row,min_col:max_col] = np.logical_or(
                img_out[z,min_row:max_row,min_col:max_col],
                img_cbbn
        )
    return img_out

def find_rings(img_objt,img_mask):
    """
    img_objt: 3d grayscale image of vacuoles
    img_mask: 2d label image of cells
    """
    img_out = np.zeros_like(img_objt,dtype=int)
    properties_cell = measure.regionprops(img_mask)
    for pcell in properties_cell:
        min_row,min_col,max_row,max_col = pcell.bbox
        img_cube = img_objt[:,min_row:max_row,min_col:max_col]
        for z in range(img_cube.shape[0]):
            img_cube[z] = img_cube[z]*pcell.image
        th_otsu  = filters.threshold_otsu(img_cube)
        img_cbbn = (img_cube > th_otsu)
        # img_cblb = measure.label(img_cbbn)
        # properties_objt = measure.regionprops(img_cblb)
        # for pobjt in properties_objt:
        #     if not bool(img_cbbn[tuple((int(coor) for coor in pobjt.centroid))]): # hollow center
        #         minz,minr,minc,maxz,maxr,maxc = pobjt.bbox
        #         img_cbbn[minz:maxz,minr:maxr,minc:maxc] = np.logical_or(
        #             img_cbbn[minz:maxz,minr:maxr,minc:maxc],
        #             fill_hole_zbyz(pobjt.image))
        img_out[:,min_row:max_row,min_col:max_col] = np.logical_or(
            img_out[:,min_row:max_row,min_col:max_col],
            img_cbbn
        )
    return img_out

def fill_hole_zbyz(img_obj):
    img_out = np.zeros_like(img_obj)
    for z in range(img_obj.shape[0]):
        img_out[z] = ndi.binary_fill_holes(img_obj[z])
        img_out[z] = morphology.binary_opening(img_out[z])
    return img_out

def wrapper(zipped):
    path_vph,path_cell,path_bin = zipped
    if not path_cell.exists():
        print(f">>>WARNING: {path_cell.name} does not exist, skipped.")
        return None
    img_vph = load_nd2_plane(str(path_vph),frame="zyx",axes="tc",idx=1)
    img_cell = io.imread(str(path_cell))

    preprocesses = [
        filters.gaussian,
        lambda x: (x-x.min())/(x.max()-x.min())
    ]
    for prep in preprocesses:
        img_vph = prep(img_vph)
    
    img_bin = find_rings_otsu_zbyz(img_vph,img_cell)
    io.imsave(str(path_bin),util.img_as_ubyte(img_bin),plugin='tifffile')
    print(f">>>INFO: {path_bin.name} finished.")
    return None

def latsingdan(process_each:callable,batch,skip:list=[],finished=None):
    """
    Batch process items in an iterable.
    INPUTS:
        process_each:callable, the function to apply to each file
        folder:str/Path
        glob:str,              file name pattern to match
        skip:iterable,         filenames (no paths) to skip
        finished:list,         to store names of finished items 
    OUPUTS:
        None,   a side-effect function
        OR finished:list, filenames (no paths) with `process_each` applied
    """
    for each in batch:
        if each in skip:
            continue
        process_each(each)
        if finished is not None:
            finished.append(each)
    return None

# MAIN:

folder_vph  = "vph/vph/unmixed"
folder_cell = "vph/cell/all"
folder_bin  = "vph/bin/otsu_zbyz"

list_blob  = [iv for iv in Path(folder_vph).glob("*blue*.nd2")]
list_cell = [Path(folder_cell)/(ic.stem.replace("unmix-blue","binCell-red")+".tif") for ic in list_blob]
list_bin = [Path(folder_bin)/(ib.stem.replace("unmix-blue","bin-vph1")+".tif") for ib in list_blob]

list_finished = []
latsingdan(
    wrapper,
    zip(list_blob,list_cell,list_bin),
    finished=list_finished
)
