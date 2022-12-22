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



# Load the image
reader = CziReader("extern_Synlab_2156_17_3_MTB.czi")
# Get whole image
smear = reader.get_image_data("MYX", C=0)

tile=smear[673]


otsu_st_16_cleaned_from_noise = otsu_cleaned_split_thresholding(tile)
cleaned_cc = clean_connected_components(otsu_st_16_cleaned_from_noise)
num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(
            np.uint8(cleaned_cc), connectivity=8)
chunks_to_ignore=ignore_connected_component_list(tile)

cleaned_stats=remove_tiles_to_ignore(stats, chunks_to_ignore)
center_of_masses=find_center_of_mass(cleaned_stats)
_5050bacilli=crop_images(tile, center_of_masses)
visualize_napari(_5050bacilli[0],"primo")

