# %%
import numpy as np
from pathlib import Path
from skimage import util,io,exposure,transform,segmentation,measure
from unet import neural_network,segment
from nd2reader import ND2Reader

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

def yeaz_label(img_i,min_dist):
    """
    Use YeaZ without GUI to segment cells in an image.
    INPUT:  
        img_i, a 2-D float image containing cells to segment. 
        min_dist, the minimum distance passed to segment()
    OUTPUT: img_o, a 2-D uint label image, same shape as img_i.
    """
    img_exposure  = exposure.equalize_adapthist(img_i)
    img_predict   = neural_network.prediction(img_exposure,False)
    img_threshold = neural_network.threshold(img_predict)
    img_segment   = segment.segment(
                                    img_threshold,img_predict,
                                    min_distance=min_dist,
                                    topology=None
                                   )
    print(img_segment.max())
    img_o = util.img_as_uint(img_segment.astype(np.int))
    return img_o

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

# customized part
transf_affine = transform.AffineTransform(matrix=np.fromfile("transform_inverse.dat").reshape((3,3)))

processes = [
    lambda x: transform.rotate(x,90),
    lambda x: transform.rescale(x,0.25),
    lambda x: transform.warp(x,transf_affine),
    lambda x: (x-x.min())/(x.max()-x.min())
]

def wrapper(zipped):
    path_file,path_out = zipped
    str_file = str(path_file)

    img_i = load_nd2_plane(str_file,frame='yx',axes='t',idx=0)
    for prep in processes:
        img_i = prep(img_i)
    
    img_b = yeaz_label(img_i,min_dist=5)
    img_b = segmentation.clear_border(img_b)
    properties = measure.regionprops(img_b)
    for prop in properties:
        if prop.area< 50: # hard coded threshold, bad
            img_b[img_b==prop.label] = 0
    img_b = measure.label(img_b)

    img_o = np.zeros((512,512),dtype=int)
    shape0,shape1 = img_b.shape
    img_o[:shape0,:shape1] = img_b

    io.imsave(str(path_out),util.img_as_uint(img_o))
    print(f">>>> finished segmenting {path_out.stem}.")
    return None

# %%
folder = r'D:\Documents\FILES\lab_RAW\2021-05-26-EYrainbow-glucose_largerBF'
files_i = list(Path(folder).glob("BFcameraAfter_EYrainbow_glu-0-5_field-7.nd2"))
files_o = [Path(r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_binCell")/(fi.stem.replace("BFcameraAfter","binCell")+".tif") for fi in files_i]
# %%
list_finish = []
latsingdan(wrapper,batch=zip(files_i,files_o),finished=list_finish)

# %%
