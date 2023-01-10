import argparse
import yaml
import sys
import numpy as np
import multiprocessing as mp

#from aicsimageio import AICSImage
#import napari
#from aicsimageio.readers import CziReader
#import cv2 as cv
#import logging  #maybe useful for debugging


#this needs specific parameters in config

from loader import Loader
from preprocess import Preprocessing
from thresholding import Thresholding
from postprocessing import Postprocessing
from cropping import cropping
from visualization import visualize_all_list_napari, add_bounding_boxes, is_blurry

def arguments_parser():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Tubercolosis Detection')
    parser.add_argument('config', type=str, default='configs/thresholding.yaml',
                        help='configure file for thresholding experiments')
    return parser

def single_tile_processing(config, tile_number):


    ####### BEGIN LOADING #######
    load_config = config['load']
    loader = Loader(load_config['czi_path'], tile_number)
    loader.load()
    img = loader.data_array
    print(f"Tile succesfully loaded, shape: {img.shape}")
    ######## END LOADING ########

    if is_blurry(img):
        pass
    else:
        ######## BEGIN PREPROCESSING ########
        preprocess_config = config['preprocessing']
        preprocess = Preprocessing(img)
        if preprocess_config['algorithm'] == "sharp":
            sharpened_img = preprocess.sharpen()
        ######## END PREPROCESSING ########

        ######## BEGIN THRESHOLDING ########
        threshold_config = config['thresholding']
        threshold = Thresholding(sharpened_img, threshold_config)
        thresholded_img = threshold.apply()
        ######## END THRESHOLDING ########

        ######## BEGIN POSTPROCESSING ########
        postprocessing_config = config['postprocessing']
        postprocess = Postprocessing(thresholded_img, postprocessing_config)
        whole_img_not_cleaned, final_image, num_bacilli = postprocess.apply()
        ######## END POSTPROCESSING ########

        ######## BEGIN ADD BOUNDING BOXES ########
        image_boxes = img.copy()
        image_boxes = add_bounding_boxes(image_boxes, final_image)

        ######## END BOUNDING BOXES ########

        ######## BEGIN CROPPING ########
        cropping_function = cropping(img, final_image)
        cropped_images = cropping_function.crop_and_pad()
        ######## END CROPPING ########

    return cropped_images, num_bacilli


def main():

    parser = arguments_parser()
    pars_arg = parser.parse_args()
    # read config as dictionary
    with open(pars_arg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    load_config = config['load']
    loader = Loader(load_config['czi_path'], load_config['tile'])
    loader.load()
    smear = loader.data_array
    num=smear.shape[0]
    del smear

    total_number_bacilli = 0
    for i in range(0, num):
        tile_cropped_images, num_bacilli = single_tile_processing(config, i)
        total_number_bacilli += num_bacilli
        if i==0:
            total_crop_images = tile_cropped_images
        else:
            total_crop_images = np.concatenate((total_crop_images, tile_cropped_images), axis=0)
    # parallelize for loop
    args= [i for i in range(0,num)]
    _5050bacilli = single_tile_processing(config,0)
    with mp.Pool() as pool:
        for result in pool.imap_unordered(config, args):
            _5050bacilli= np.concatenate((_5050bacilli, result), axis=0)

if __name__ == '__main__':
    main()

