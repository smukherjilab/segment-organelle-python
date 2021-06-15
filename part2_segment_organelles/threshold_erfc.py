# %%
import numpy as np
from pathlib import Path
from nd2reader import ND2Reader
from scipy import special
from skimage import measure,util,io,filters

def get_nd2_size(path:str):
    with ND2Reader(path) as images:
        size = images.sizes
    return size

def load_nd2_plane(path:str,frame:str,axes:str,idx:int=0):
    """read an image flexibly with ND2Reader."""
    with ND2Reader(path) as images:
        images.bundle_axes = frame
        images.iter_axes = axes
        img = images[idx]
    return img.squeeze()

def latsingdan(process_each:callable,batch,skip:list=[],finished=None):
    """
    Batch process items in an iterable.
    INPUTS:
        process_each:callable, the function to apply to each file
        batch:iterable,        
        skip:list,             items in batch to skip
        finished:list,         to store names of finished items 
    OUPUTS:
        None,   a side-effect function
        (Optional, side effect) finished:list, filenames (no paths) with `process_each` applied
    """
    for each in batch:
        if each in skip:
            continue
        process_each(each)
        if finished is not None:
            finished.append(each)
    return 

def bkgd_likelihood(img_obj,img_mask):
    """
    INPUTS:
        img_obj:  a 3d1c image of the objects (perhaps will be 2d)
        img_mask: a 2d label image indicating the foregrounds
    OUTPUT:
        img_out: an image same size of img_obj, the likelyhood of the pixel NOT background.
    """
    img_out = np.zeros_like(img_obj)
    img_bkgd = (img_mask>0)
    for i,slice_obj in enumerate(img_obj):
        img_ma = np.ma.masked_array(slice_obj,mask=img_bkgd)
        
        data_mask = img_ma.flatten()
        mean_mask = data_mask.mean()
        stdv_mask = data_mask.std()

        img_norm = (slice_obj - mean_mask)/(15*stdv_mask)
        img_out[i] = special.erf(np.abs(img_norm)/np.sqrt(2))
    return img_out

def open_organelle(path:str,organlle:str):
    if   organlle == 'pex3':
        return load_nd2_plane(str(path),frame="zyx",axes='tc',idx=0)
    elif organlle == 'vph1':
        return load_nd2_plane(str(path),frame="zyx",axes='tc',idx=1)
    elif organlle == "sec61":
        return load_nd2_plane(str(path),frame="zyx",axes='t',idx=0)
    elif organlle == 'sec7':
        return np.sum(load_nd2_plane(str(path),frame="czyx",axes='t',idx=0),axis=0)
    elif organlle == 'tom70':
        # return load_nd2_plane(str(path),frame="zyx",axes='tc',idx=0)
        return io.imread(str(path))[0]
    elif organlle == 'erg6':
        # return load_nd2_plane(str(path),frame="zyx",axes='tc',idx=1)
        return io.imread(str(path))[1]
    else:
        raise ValueError("Unrecognized organelle type.")

def wrapper_test(zipped):
    path_i,path_c,path_o,orga,thsh = zipped
    img_i = open_organelle(str(path_i),organlle=orga)
    img_c = io.imread(str(path_c))
    
    img_i = filters.gaussian(img_i)
    img_o = bkgd_likelihood(img_i,img_c)
    img_o = (img_o>thsh)

    io.imsave(str(path_o),util.img_as_ubyte(img_o))
    print(f">>>INFO: {path_o.name} finished.")
    
def wrapper_work():
    raise NotImplementedError

# %%
path_cell = Path("i/binCell_EYrainbow_glu-200_field-1.tif")
list_files = [
    Path("i/unmixed-blue_EYrainbow_glu-200_field-1.nd2"),
    Path("i/unmixed-blue_EYrainbow_glu-200_field-1.nd2"),
    Path("i/spectral-green_EYrainbow_glu-200_field-1.nd2"),
    Path("i/spectral-yellow_EYrainbow_glu-200_field-1.nd2"),
    Path("i/rollball-red_EYrainbow_glu-200_field-1.tif"),
    Path("i/rollball-red_EYrainbow_glu-200_field-1.tif")]
list_orga = ['pex3','vph1','sec61','sec7','tom70','erg6']
list_thsh = [0.75,0.6,0.5,0.65,0.5,0.8]

list_params = []
for y,path in enumerate(list_files):
    list_params.append((
        path,
        path_cell,
        Path('o')/("bkgd15bw-"+list_orga[y]+"_"+path.stem.partition("_")[2]+".tif"),
        list_orga[y],
        list_thsh[y]
    ))
# %%
finished = []
latsingdan(wrapper_test,list_params,finished=finished)

# %%
list_orga = ['pex3','vph1','sec61','sec7','tom70','erg6']
dict_folderidx = {
    'pex3':  0,
    'vph1':  0,
    'sec61': 0,
    'sec7':  0,
    'tom70': 1,
    'erg6':  1
}
dict_thresholds = {
    'pex3':  0.75,
    'vph1':  0.6,
    'sec61': 0.5,
    'sec7':  0.65,
    'tom70': 0.5,
    'erg6':  0.8
}
dict_prefix = {
    'pex3':  "unmixed-blue_",
    'vph1':  "unmixed-blue_",
    'sec61': "spectral-green_",
    'sec7':  "spectral-yellow_",
    'tom70': 'rollball-red_',
    'erg6':  'rollball-red_'
}
dict_suffix = {
    'pex3':  ".nd2",
    'vph1':  ".nd2",
    'sec61': ".nd2",
    'sec7':  ".nd2",
    'tom70': ".tif",
    'erg6':  ".tif"
}
list_folder_orga = [
    r"D:\Documents\FILES\lab_RAW\2021-05-26-EYrainbow-glucose_largerBF",
    r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_redRollingBall"
]
folder_cell = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_binCell"
folder_otpt = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_binOrganelle"

list_params = []
for path_cell in Path(folder_cell).glob("*.tif"):
    stem_cell = path_cell.stem.partition("_")[2]
    for orga in list_orga:
        path_orga = Path(list_folder_orga[dict_folderidx[orga]])/"".join([
                dict_prefix[orga],
                stem_cell,
                dict_suffix[orga]
            ])
        path_otpt = Path(folder_otpt)/("".join([
            "bin-",orga,"_",
            path_orga.stem.partition("_")[2],
            ".tif"
        ]))
        list_params.append((
            path_orga,
            path_cell,
            path_otpt,
            orga,
            dict_thresholds[orga]
        ))
# %%
finished = []
latsingdan(wrapper_test,list_params,finished=finished)

# %%
