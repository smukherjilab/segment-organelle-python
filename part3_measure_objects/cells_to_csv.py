import pandas as pd
from pathlib import Path
from skimage import io,measure

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

def parse_stem_cell(stem:str):
    return {
        "experiment": "glucose",
        "condition": stem.partition("glu-")[2].partition("_")[0].replace("-","."),
        "hour": 3,
        "field": stem.partition("field-")[2]
    }

def cell2csv(zipped):
    path_img,path_csv = zipped
    img_cell = io.imread(str(path_img))
    parsed = parse_stem_cell(path_img.stem)
    measured = measure.regionprops_table(
        img_cell,
        properties = ('label','area','centroid','bbox','eccentricity')
    )
    dict_write = parsed | measured
    df_write =  pd.DataFrame(dict_write)
    df_write.rename(columns={'label':'label-cell'},inplace=True)
    df_write.to_csv(
        str(path_csv),
        index=False
    )    
    return None

folder_i = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_binCell"
folder_o = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_csv\cells"

files_i = list(Path(folder_i).glob("*.tif"))
files_o = [Path(folder_o)/(i_file.stem.replace("bin-blue","cell")+".csv") for i_file in files_i]

list_finished = []
latsingdan(cell2csv,zip(files_i,files_o),finished=list_finished)


