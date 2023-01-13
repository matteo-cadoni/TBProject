import argparse
import yaml
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#from aicsimageio import AICSImage
#import napari
#from aicsimageio.readers import CziReader
#import cv2 as cv
#import logging  #maybe useful for debugging

from loader import Loader
from preprocess import Preprocessing
from thresholding import Thresholding
from postprocessing import Postprocessing
from cropping import Cropping
from interactivelabelling import InteractiveLabeling
from interactive_config import InteractiveConfig, change_yaml

from visualization import visualize_all_list_napari, add_bounding_boxes, is_blurry

def arguments_parser():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Tubercolosis Detection')
    parser.add_argument('config', type=str, default='configs/thresholding.yaml',
                        help='configure file for thresholding experiments')
    return parser

def main():
    parser = arguments_parser()
    pars_arg = parser.parse_args()
    
    # read config as dictionary
    with open(pars_arg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ####### BEGIN INTERACTIVE CONFIGURATION #######
    i_c = InteractiveConfig()
    configurations = i_c.run()
    config = change_yaml(configurations, config)
    ####### END INTERACTIVE CONFIGURATION #######

    ####### BEGIN LOADING #######
    load_config = config['load']
    loader = Loader(load_config['czi_path'], load_config['tile'])
    loader.load()
    if load_config['tile'] == 'None':
        smear = loader.data_array
        print(f"Smear succesfully loaded, shape: {smear.shape}")
        total_number_bacilli = 0
    else:
        img = loader.data_array
        print(f"Tile succesfully loaded, shape: {img.shape}")
    ######## END LOADING ########
    if load_config['tile'] == 'None':
        for tiles in range(0, smear.shape[0]):
            img = smear[tiles, :, :]
            print("Tile: ", tiles)
            if load_config['blurry_deselection']:
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
                    not_cleaned_img, final_image, num_bacilli = postprocess.apply()
                    total_number_bacilli += num_bacilli
                    ######## END POSTPROCESSING ########

                    ######## BEGIN CROPPING ########
                    cropping_config = postprocessing_config['crop']
                    cropping_function = Cropping(img, final_image)
                    if cropping_config['crop']:
                        if 'cropped_img' in locals():
                            cropped_img = cropping_function.crop()
                        else:
                            cropped_img = np.concatenate((cropped_img, cropping_function.crop()), axis=0)            #might need square brackets around cropping_function.crop() to make it work

                     ######## END CROPPING ########
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
                not_cleaned_img, final_image, num_bacilli = postprocess.apply()
                total_number_bacilli += num_bacilli
                ######## END POSTPROCESSING ########

                ######## BEGIN CROPPING ########
                cropping_config = postprocessing_config['crop']
                cropping_function = Cropping(img, final_image)
                if cropping_config['crop']:
                    if tiles == 0:
                        cropped_img = cropping_function.crop()
                    else:
                        cropped_img = np.concatenate((cropped_img, cropping_function.crop()), axis=0)  # might need square brackets around cropping_function.crop() to make it work
                ######## END CROPPING ########
        print("Total number of bacilli: ", total_number_bacilli)  
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
        cropping_config = postprocessing_config['crop']
        cropping_function = Cropping(img, final_image)
        if cropping_config:
            cropped_images=cropping_function.crop_and_pad()

        ######## END CROPPING ########

        ######## BEGIN INTERACTIVE LABELING #######

        i_l=InteractiveLabeling(cropped_images)
        labels= i_l.run()

        ######### END INTERACTIVE LABELING ########

        ######## BEGIN DATASET CREATION ########
        dataframe = pd.DataFrame()
        for i in range(0, labels.shape[0]):
            d = {'image': [cropped_images[i]], 'label': [labels[i]]}
            df2 = pd.DataFrame(d)
            dataframe = pd.concat([dataframe, df2], ignore_index=True)
        ######## END DATASET CREATION ########

        ######## BEGIN VISUALIZATION ########
        visualization_config = config['visualization']
        show = visualization_config['show']
        if show:
            images = [img, sharpened_img, whole_img_not_cleaned, final_image, image_boxes]
            strings_names = ['img', 'sharpened_img', 'whole_img_not_cleaned', 'final_image','image_boxes']
            visualize_all_list_napari(images, strings_names)
        #currently opening the napari visualizer stops the execution of the code
        ######## END VISUALIZATION ########
        
if __name__ == "__main__":
    main()
