from aicsimageio import AICSImage
import napari
from aicsimageio.readers import CziReader
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Load the image
reader = CziReader("extern_Synlab_2156_17_3_MTB.czi")
# Get whole image
smear = reader.get_image_data("MYX", C=0)

#delete blurry images
# detect blurry image based on fourier transform



cleaned_smear=get_not_blurry_tiles(smear)

#delete smear from memory
del smear

#get 50 x 50 bacilligit a
def get_5050bacilli(images_list):
    _5050bacilli=[]
    for tile in images_list:
        otsu_st_16_cleaned_from_noise = otsu_cleaned_split_thresholding(img)
        cleaned_cc = clean_connected_components(otsu_st_16_cleaned_from_noise)
        # get stats for drawing boxes
        num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(
            np.uint8(cleaned_cc), connectivity=8)
        center_of_masses=find_center_of_mass( stats)
        _5050bacilli.append(crop_images(tile, center_of_masses))
    return _5050bacilli