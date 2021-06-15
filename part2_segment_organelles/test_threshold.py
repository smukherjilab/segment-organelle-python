# %%
import numpy as np
from pathlib import Path
from nd2reader import ND2Reader
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

def segment_by_percentile(img,percent,mask=None):
    if mask is not None:
        img = np.ma.masked_array(img,mask=mask)
    img_bin = np.zeros_like(img,dtype=bool)
    for z in range(img.shape[0]):
        threshold = np.percentile(img[z],100-percent)
        img_bin[z] = (img[z]>threshold)
    return img_bin

def segment_bypercentile_cellwise(img_objt,img_cell,percent):
    """
    img_objt: 3d grayscale image of vacuoles
    img_mask: 2d label image of cells
    """
    img_out = np.zeros_like(img_objt,dtype=int)
    properties_cell = measure.regionprops(img_cell)
    for pcell in properties_cell:
        min_row,min_col,max_row,max_col = pcell.bbox
        img_cube = img_objt[:,min_row:max_row,min_col:max_col]
        for z in range(img_cube.shape[0]):
            img_cube[z] = img_cube[z]*pcell.image
        img_out[:,min_row:max_row,min_col:max_col] = np.logical_or(
            img_out[:,min_row:max_row,min_col:max_col],
            segment_by_percentile(img_cube,percent)
        )
    return img_out

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
    return None

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
        return load_nd2_plane(str(path),frame="zyx",axes='tc',idx=0)
        # return io.imread(str(path))[0]
    elif organlle == 'erg6':
        return load_nd2_plane(str(path),frame="zyx",axes='tc',idx=1)
        # return io.imread(str(path))[1]
    else:
        raise ValueError("Unrecognized organelle type.")

def wrapper(zipped):
    path_i,path_o,orga,threshold = zipped
    img = open_organelle(str(path_i),organlle=orga)
    processes = [
        filters.gaussian,
        lambda x: (x-x.min())/(x.max()-x.min()),
        lambda x: segment_by_percentile(x,threshold)
    ]
    for proc in processes:
        img = proc(img)
    io.imsave(str(path_o),util.img_as_ubyte(img))
    print(f">>>INFO: {path_o.name} finished.")
    
# %%
# Part 0: test threhold by cell for vph1, does not work
def wrapper_cellwise(zipped):
    path_i,path_c,path_o,orga,threshold = zipped
    img = open_organelle(str(path_i),organlle=orga)
    img_c = io.imread(str(path_c))
    processes = [
        filters.gaussian,
        lambda x: (x-x.min())/(x.max()-x.min())
    ]
    for proc in processes:
        img = proc(img)
    img = segment_bypercentile_cellwise(img,img_c,threshold)
    io.imsave(str(path_o),util.img_as_ubyte(img))
    print(f">>>INFO: {path_o.name} finished.")

list_threshold = [i/1. for i in range(1,22)]
list_param_cell = [
    (
        Path("i/unmix-blue_EYrainbow_glu-100_field-1.nd2"),
        Path("i/binCell-red_EYrainbow_glu-100_field-1.tif"),
        Path(f"o/cellwise-{str(th).replace('.','-')}_EYrainbow_glu-100_field-1.tif"),
        "vph1",
        th
    ) for th in list_threshold
]

list_finish_cell = []
latsingdan(wrapper_cellwise,list_param_cell,finished=list_finish_cell)

# %%
# Part 1: find proper thresholds
list_threshold = [i/10. for i in range(1,31)]
list_files = [
    Path("i/unmixed-blue_EYrainbow_glu-200_field-1.nd2"),
    Path("i/unmixed-blue_EYrainbow_glu-200_field-1.nd2"),
    Path("i/spectral-green_EYrainbow_glu-200_field-1.nd2"),
    Path("i/spectral-yellow_EYrainbow_glu-200_field-1.nd2"),
    Path("i/unmixed-red_EYrainbow_glu-200_field-1.nd2"),
    Path("i/unmixed-red_EYrainbow_glu-200_field-1.nd2")]
list_orga = ['pex3','vph1','sec61','sec7','tom70','erg6']

list_params = []
for x in list_threshold:
    for y,path in enumerate(list_files):
        list_params.append((
            path,
            Path('o')/(list_orga[y]+"_bin-"+str(x).replace(".","-")+"_"+path.stem.partition("_")[2]+".tif"),
            list_orga[y],
            x
        ))

finished = []
latsingdan(wrapper,list_params,finished=finished)

# %%
# Part 2.0: batch run tom70 and erg6
dict_thresholds = {
    'pex3':  0.1,
    'sec61': 1.0,
    'sec7':  0.1,
    'erg6':  0.1,
    'tom70': 2.0
}
folder_red = r"D:\Documents\FILES\lab_RAW\2021-03-11_EYrainbow_Leucine_unmixed"
list_files_red = list(Path(folder_red).glob("*.nd2"))
folder_bin = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY279xSixColor_Leucine_LargerExperiment\EY279xSixColor_Leucine_LargerDataset_binOrganelle"
params_red = []
list_skip = []
for red in ["erg6","tom70"]:
    for file_red in list_files_red:
        params_red.append((
            file_red,
            Path(folder_bin)/(file_red.stem.replace("red","bin-"+red)+".tif"),
            red,
            dict_thresholds[red]
        ))
red_finished = []
latsingdan(wrapper,params_red,skip=list_skip,finished=red_finished)

# %%
# Part 3.0: run pex3, sec61, and erg6
folders_rest = [
    r"D:\Documents\FILES\lab_RAW\2021-02-21_EYrainbow_leu-0_hour-24",
    r"D:\Documents\FILES\lab_RAW\2021-02-20_EYrainbow_leu-25_hour-24",
    r"D:\Documents\FILES\lab_RAW\2021-02-20_EYrainbow_leu-0_hour-3",
    r"D:\Documents\FILES\lab_RAW\2021-02-19_EYrainbow_leu-25_hour-3",
    r"D:\Documents\FILES\lab_RAW\2021-02-18_EYrainbow_leu-50_hour-24",
    r"D:\Documents\FILES\lab_RAW\2021-02-18_EYrainbow_leu-50_hour-3",
    r"D:\Documents\FILES\lab_RAW\2021-02-16_EYrainbow_leu-75_hour-24",
    r"D:\Documents\FILES\lab_RAW\2021-02-15_Eyrainbow_leu-75_hour-3",
    r"D:\Documents\FILES\lab_RAW\2021-02-12_EYrainbow_leu-100_hour-24",
    r"D:\Documents\FILES\lab_RAW\2021-02-11_EYrainbow_leu-100_hour-3"
]
prefix_rest = {
    'pex3':  "blue",
    'sec61': "green",
    'erg6':  "yellow"
}
params_rest = []
for rest in ['pex3','sec61','erg6']:
    for folder_rest in folders_rest:
        for file_rest in Path(folder_rest).glob(prefix_rest[rest]+"*"):
            params_rest.append((
                file_rest,
                Path(folder_bin)/(file_rest.stem.replace(prefix_rest[rest],"bin-"+rest)+".tif"),
                rest,
                dict_thresholds[rest]
            ))
latsingdan(wrapper,params_rest,finished=finished)

# %%
# sec7 redo and threshold slice by slice.
params_sec7 = []
for file_rest in Path(folder_red).glob("*.nd2"):
    params_sec7.append((
        file_rest,
        Path(folder_bin)/(file_rest.stem.replace("red","bin-sec7")+".tif"),
        "sec7",
        0.1
    ))
finished_sec7 = []
latsingdan(wrapper,params_sec7,finished=finished_sec7)

# %%
path_test = "i/rollball-red_EYrainbow_glu-0_field-4.tif"
img_test = io.imread(path_test)
# %%
# 2021-05-25 Ey2796SixColors-Glucose
list_organeles = ["pex3","sec61","sec7","tom70","erg6"]
folder_cell = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose\EY289xSixColor_Glucose_binCell"
folder_biny = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose\EY289xSixColor_Glucose_binOrganelle"
list_folders = [
    r"D:\Documents\FILES\lab_RAW\2021-05-10_EYrainbow-glucose-unmixed",
    r"D:\Documents\FILES\lab_RAW\2021-03-29_Eyrainbow-glucose",
    r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose\EY289xSixColor_Glucose_rollingballRed"
]
dict_thresholds = {
    'pex3':  0.1,
    'sec61': 1.0,
    'sec7':  0.1,
    'erg6':  0.1,
    'tom70': 2.0
}
dict_folders = {
    'pex3':  0,
    'sec61': 1,
    'sec7':  1,
    'tom70': 2,
    'erg6':  2
}
dict_colors = {
    'pex3':  "blue",
    'sec61': "green",
    'sec7':  "yellow",
    'tom70': "red",
    'erg6':  "red"
}

params = []
for orga in list_organeles:
    glb = "".join(["*",dict_colors[orga],"*"])
    for path_orga in Path(list_folders[dict_folders[orga]]).glob(glb):
        name_orga = path_orga.stem
        file_orga = path_orga
        name_biny = f"bin-{orga}_" + name_orga.partition("_")[2] + ".tif"
        file_biny = Path(folder_biny)/name_biny
        params.append((file_orga,file_biny,orga,dict_thresholds[orga]))

# %%
finish_5colors = []
latsingdan(wrapper,params,finished=finish_5colors)
# %%
