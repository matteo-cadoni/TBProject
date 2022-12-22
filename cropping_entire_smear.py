from aicsimageio import AICSImage

import aicspylibczi

import napari
from aicsimageio.readers import CziReader
import cv2 as cv
import numpy as np
from split_otsu import *
from cleaning_and_cc import *
from cropping import *
from visualization import *
from blurry_deselection import *
from single_tile_cropping import *
import multiprocessing as mp

# Load the image
reader = CziReader("extern_Synlab_2156_17_3_MTB.czi")
# Get whole image
smear = reader.get_image_data("MYX", C=0)
cleaned_smear=get_not_blurry_tiles(smear)
del smear


#for tile in smear:
    #intermediate=get_crops_one_tile(tile)
    #_5050bacilli = _5050bacilli.append(intermediate)

_5050bacilli=[]
with mp.Pool() as pool:
    for result in pool.imap_unordered(get_crops_one_tile,cleaned_smear):
        _5050bacilli.append(result)