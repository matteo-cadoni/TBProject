import argparse
import yaml
import sys

#from aicsimageio import AICSImage
#import napari
#from aicsimageio.readers import CziReader
#import cv2 as cv
#import logging  #maybe useful for debugging

from loader import Loader
from preprocess import Preprocessing
from thresholding import Thresholding
from postprocessing import Postprocessing
from visualization import visualize_all_list_napari

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
    
    ####### BEGIN LOADING #######
    load_config = config['load']
    loader = Loader(load_config['czi_path'], load_config['tile'])
    loader.load()
    smear = loader.data_array
    print(f"Smear succesfully loaded, shape: {smear.shape}")
    total_number_bacilli = 0
    ######## END LOADING ########
    for tiles in range(0, smear.shape[0]):
        img = smear[tiles, :, :]
        print("Tile: ", tiles)

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
        final_image, num_bacilli = postprocess.apply()
        total_number_bacilli += num_bacilli
        ######## END POSTPROCESSING ########     
    print("Total number of bacilli: ", total_number_bacilli)
    ######## BEGIN VISUALIZATION ########
    #images = [img, sharpened_img, final_image]
    #strings_names = ['img', 'sharpened_img', 'final_image']
    #visualize_all_list_napari(images, strings_names)
    #currently opening the napari visualizer stops the execution of the code
    ######## END VISUALIZATION ########
    
if __name__ == "__main__":
    main()
