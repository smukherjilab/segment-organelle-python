# %%
import numpy as np
from pathlib import Path
from nd2reader import ND2Reader
from skimage import io,util,measure,exposure
from unet import neural_network,segment


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

norm_mean_std = lambda x: (x-x.mean())/x.std()
norm_max_min  = lambda x: (x-x.min())/(x.max()-x.min())
# %% [markdown] 
# This file tries to find a better way to get the cell boundaries from several 
# multi-channel images of different organelles and use YeaZ neural network.
# 
# We have:
# - 4 different colors: blue, green, yellow, red; 2 unmixed images: blue, red
# - n channels for each color: {blue:5,green:1,yellow:5,red:6,unmixed:2}
# - 31 z slices for each channel in each images
# 
# 
# The choices we can make:
# - The order to combine the xy planes: channel-color-z channel-z-color or z-
# - The way to combine: raw/normalize max-min/normalize mea-std, mean(sum)/max
# - Time to use YeaZ
# - Parameter of YeaZ: min_dist
# 
# 
# Therefore:
# - norm_max_min(   mean(mean(norm_mean_std(img[iczyx],yx),ic),z))
# - norm_max_min(slice15(mean(norm_mean_std(img[iczyx],yx),ic),z))
# %%
# This is an ancient code block.
def exclude_movers(img1,img2):
    """
    Remove the cells in img1 whose centroids are not in img2.
    INPUTS:
        img1,img2: label images of the same field of view from 2 spectral detectors.
    OUTPUT:
        img0: a label image whose cell centroids are in both images.
    """
    img0 = np.zeros_like(img1,dtype=np.uint16)
    properties = measure.regionprops(img1)
    count = 0
    for prop in properties:
        if img2[prop.centroid] > 0:
            count += 1
            img0[prop.coords] = count
    return img0

# %%
img_i = load_nd2_plane("test/i/spectral-green_EYrainbow_glu-100_field-1.nd2",frame='zyx',axes='t',idx=0)
# %%
for i in range(31):
    print(img_i[i].mean(),img_i[i].std())
# %%
img_norm_alle = norm_mean_std(img_i)
img_norm_each = np.zeros_like(img_i)
for i in range(img_i.shape[0]):
    img_norm_each[i] = norm_mean_std(img_i[i])

# %%
img_sum_alle = np.sum(img_norm_alle,axis=0)
img_sum_each = np.sum(img_norm_each,axis=0)
# %%
img_io_alle = norm_max_min(img_sum_alle)
img_io_each = norm_max_min(img_sum_each)
# %%
io.imsave("test/o/green-alle.tif",util.img_as_float(img_io_alle))
io.imsave("test/o/green-each.tif",util.img_as_float(img_io_each))
# %%
threshold = np.percentile(img_sum_alle,99)
img_cutoff = np.copy(img_sum_alle)
img_cutoff[img_sum_alle>threshold] = threshold
img_io_cutoff = norm_max_min(img_cutoff)
# %%
img_yeaz = yeaz_label(img_io_cutoff,min_dist=5)
# %%
io.imsave("test/o/yeaz_cutoff99.tif",util.img_as_uint(img_yeaz))
# %%
