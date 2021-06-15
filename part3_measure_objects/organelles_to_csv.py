import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,measure

def parse_stem_organelle(stem:str):
    return {
        "experiment": "glucose",
        "condition": stem.partition("glu-")[2].partition("_")[0].replace("-","."),
        "hour": 3,
        "field": stem.partition("field-")[2],
        "organelle": stem.partition("bin-")[2].partition("_")[0]
    }

def organelle2csv(zipped):
    path_orga,path_cell,path_csv = zipped
    parsed = parse_stem_organelle(path_orga.stem)
    img_cell = io.imread(str(path_cell))
    img_orga = io.imread(str(path_orga))
    measured_cell = measure.regionprops(img_cell)
    dfs_orga = []
    for cell in measured_cell:
        parsed["label-cell"] = cell.label
        min_row, min_col, max_row, max_col = cell.bbox
        img_orga_crop = img_orga[:,min_row:max_row,min_col:max_col]
        img_cell_crop = cell.image
        for z in range(img_orga_crop.shape[0]):
            img_orga_crop[z] = img_orga_crop[z]*img_cell_crop
        if not parsed["organelle"]=="sec61":
            img_orga_crop = measure.label(img_orga_crop)
        measured_orga = measure.regionprops_table(
                                                  img_orga_crop,
                                                  properties=('label','area','bbox_area','bbox')
                        )
        dict_orga = parsed | measured_orga
        dfs_orga.append(pd.DataFrame(dict_orga))
    df_orga = pd.concat(dfs_orga,ignore_index=True)
    df_orga.rename(columns={'label':'label-orga',"area":"volume-pixel",'bbox_area':'volume-bbox'},inplace=True)
    df_orga.to_csv(
        str(path_csv),
        index=False
    )
    print(f">>> finished {path_csv.stem}.")
    return None

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


folder_cell = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_binCell"
folder_orga = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_binOrganelle"
folder_csvs = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_csv\organelles"

files_orga = list(Path(folder_orga).glob("*.tif"))
files_cell = [Path(folder_cell)/(i_cell.name.replace(i_cell.name.partition("_")[0],"binCell")) for i_cell in files_orga]
files_csvs = [Path(folder_csvs)/(i_csv.stem.replace("bin-","")+".csv") for i_csv in files_orga]

list_finished=[]
latsingdan(organelle2csv,zip(files_orga,files_cell,files_csvs),finished=list_finished)