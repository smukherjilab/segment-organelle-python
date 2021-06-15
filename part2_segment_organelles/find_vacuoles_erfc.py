# %%
import numpy as np
from scipy import special
from pathlib import Path
from nd2reader import ND2Reader
from skimage import util,io,filters,morphology,measure

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

def bkgd_likelihood_2d(img_obj,img_mask):
    img_out = np.zeros_like(img_obj)
    img_bkgd = (img_mask>0)
    
    img_ma = np.ma.masked_array(img_obj,mask=img_bkgd)
    
    data_mask = img_ma.flatten()
    mean_mask = data_mask.mean()
    stdv_mask = data_mask.std()

    img_norm = (img_obj - mean_mask)/stdv_mask
    img_out = special.erfc(np.abs(img_norm)/np.sqrt(2))
    return img_out

def bkgd_likelihood(img_obj,img_mask):
    """
    INPUTS:
        img_obj:  a 3d1c image of the objects (perhaps will be 2d)
        img_mask: a 2d label image indicating the foregrounds
    OUTPUT:
        img_out: an image same size of img_obj, the likelyhood of the pixel to be background.
    """
    img_out = np.zeros_like(img_obj)
    img_bkgd = (img_mask>0)
    for i,slice_obj in enumerate(img_obj):
        img_ma = np.ma.masked_array(slice_obj,mask=img_bkgd)
        
        data_mask = img_ma.flatten()
        mean_mask = data_mask.mean()
        stdv_mask = data_mask.std()

        img_norm = (slice_obj - mean_mask)/stdv_mask
        img_out[i] = special.erfc(np.abs(img_norm)/np.sqrt(2))
    return img_out

def find_blob(img_binary,img_convex=None):
    if img_convex is None:
        img_convex = morphology.convex_hull_image(img_binary)
    img_xor = np.logical_xor(img_binary,img_convex)
    img_open = morphology.binary_opening(img_xor)
    return img_open

# working function
def find_vacuoles(img_vph,img_cell,threshold):
    img_likelihood = bkgd_likelihood(img_vph,img_cell)
    img_likely = (img_likelihood>threshold) 
    img_vph_invert = np.invert(img_likely)
    img_vph = morphology.binary_opening(img_vph_invert)

    properties = measure.regionprops(img_cell)
    img_vph_blob = np.zeros_like(img_vph,dtype=bool)
    for prop in properties:
        min_row, min_col, max_row, max_col = prop.bbox
        if (max_row-min_row<2) or (max_col-min_col<2):
            continue
        # print(prop.label,min_row, min_col, max_row, max_col)
        for z in range(img_vph.shape[0]):
            # print(z)
            img_vph_z = img_vph[z,min_row:max_row,min_col:max_col]
            img_and_z = np.logical_and(img_vph_z,prop.image)
            img_blob_z = find_blob(img_and_z)
            img_vph_blob[z,min_row:max_row,min_col:max_col] = img_blob_z
    return img_vph_blob

# break `find_vacuoles()` into 2 parts to check off focal planes.
def find_vacuoles_broken(img_vph,img_cell,threshold):
    img_likelihood = bkgd_likelihood(img_vph,img_cell)
    img_likely = (img_likelihood>threshold) # hard coded threshold, bad
    img_vph_invert = np.invert(img_likely)
    img_vph = morphology.binary_opening(img_vph_invert)
    return img_vph

def find_vacuoles_complete(img_vph,img_cell):
    properties = measure.regionprops(img_cell)
    img_vph_blob = np.zeros_like(img_vph,dtype=bool)
    for prop in properties:
        min_row, min_col, max_row, max_col = prop.bbox
        if (max_row-min_row<2) or (max_col-min_col<2):
            continue
        # print(prop.label,min_row, min_col, max_row, max_col)
        for z in range(img_vph.shape[0]):
            # print(z)
            img_vph_z  = img_vph[z,min_row:max_row,min_col:max_col]
            img_and_z  = np.logical_and(img_vph_z,prop.image)
            img_blob_z = find_blob(img_and_z)
            img_vph_blob[z,min_row:max_row,min_col:max_col] = img_blob_z
    return img_vph_blob
# end break

# %%
# pack up for iteration:
# 2021-05-07 leucine larger dataset
def packup_vph_1(zipped):
    path_vph,path_cell,path_blob = zipped
    if not path_cell.exists():
        print(f">>>WARNING: {path_cell.name} does not exist, skipped.")
        return None
    img_vph = load_nd2_plane(str(path_vph),frame="zcyx",axes="t",idx=0)
    img_cell = io.imread(str(path_cell))    

    preprocesses = [
        lambda x: np.sum(x,axis=0),
        filters.gaussian,
        lambda x: (x-x.min())/(x.max()-x.min())
    ]
    for prep in preprocesses:
        img_vph = prep(img_vph)
    
    img_blob = find_vacuoles_broken(img_vph,img_cell,threhold=10**(-3))
    name_blob = str(path_blob)
    io.imsave(name_blob,util.img_as_ubyte(img_blob))
    print(f">>>INFO: {name_blob.name} finished.")
    return None

def packup_vph_2(zipped):
    path_blob,path_cell,path_bin = zipped
    if not path_cell.exists():
        print(f">>>WARNING: {path_cell.name} does not exist, skipped.")
        return None
    img_vph  = io.imread(str(path_blob))
    img_cell = io.imread(str(path_cell))

    img_bin = find_vacuoles_complete(img_vph,img_cell)
    io.imsave(str(path_bin),util.img_as_ubyte(img_bin))
    print(f">>>INFO: {path_bin.name} finished.")
    return None
# %%
# 2021-05-13 glucose dataset
def packup_vph(zipped):
    path_vph,path_cell,path_bin = zipped
    if not path_cell.exists():
        print(f">>>WARNING: {path_cell.name} does not exist, skipped.")
        return None
    img_vph = load_nd2_plane(str(path_vph),frame="czyx",axes="t",idx=0)
    img_cell = io.imread(str(path_cell))
    
    preprocesses = [
        lambda x: np.sum(x,axis=0),
        filters.gaussian,
        lambda x: (x-x.min())/(x.max()-x.min())
    ]
    for prep in preprocesses:
        img_vph = prep(img_vph)
    
    img_bin = find_vacuoles(img_vph,img_cell,threshold=0.65)
    io.imsave(str(path_bin),util.img_as_ubyte(img_bin),plugin='tifffile')
    print(f">>>INFO: {path_bin.name} finished.")
    return None

# end pack up

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
    """
    for each in batch:
        if each in skip:
            continue
        process_each(each)
        if finished is not None:
            finished.append(each)
    return None

# %%
# MAIN:
# folder_vph  = "vph/vph/spectral"
# folder_cell = "vph/cell/dist3"
# folder_bin  = "vph/bin"

folder_vph = r"D:\Documents\FILES\lab_RAW\2021-03-29_Eyrainbow-glucose"
folder_cell = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose\EY289xSixColor_Glucose_binCell"
folder_bin = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose\EY289xSixColor_Glucose_binOrganelle"


list_vph1  = [iv for iv in Path(folder_vph).glob("*blue*.nd2")]
list_cell = [Path(folder_cell)/(ic.stem.replace("spectral-blue","binCellGreenClrBdr")+".tif") for ic in list_vph1]
list_bin = [Path(folder_bin)/(ib.stem.replace("spectral-blue","bin-vph1")+".tif") for ib in list_vph1]

# %%
list_finished = []
latsingdan(packup_vph,zip(list_vph1,list_cell,list_bin),finished=list_finished)

# %%
