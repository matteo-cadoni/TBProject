from split_otsu import *
from cleaning_and_cc import *
from cropping import *
from visualization import *
import cv2 as cv

def get_crops_one_tile(tile):
    otsu_st_16_cleaned_from_noise = otsu_cleaned_split_thresholding(tile)
    cleaned_cc = clean_connected_components(otsu_st_16_cleaned_from_noise)
    num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(
        np.uint8(cleaned_cc), connectivity=8)
    chunks_to_ignore = ignore_connected_component_list(tile)

    cleaned_stats = remove_tiles_to_ignore(stats, chunks_to_ignore)
    center_of_masses = find_center_of_mass(cleaned_stats)
    _5050bacilli = crop_images(tile, center_of_masses)
    return _5050bacilli
