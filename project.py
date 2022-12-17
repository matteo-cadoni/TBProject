import argparse
import yaml
import sys

#from aicsimageio import AICSImage
#import napari
#from aicsimageio.readers import CziReader
#import cv2 as cv
#import logging  #maybe useful for debugging

from visualization import Visualizer
from loader import Loader
from preprocess import Preprocessing
from thresholding import Thresholding

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
    img = loader.data_array
    print(f"Image succesfully loaded, shape: {img.shape}")
    ######## END LOADING ########
    
    ######## BEGIN PREPROCESSING ########
    preprocess_config = config['preprocessing']
    split = preprocess_config['split']
    preprocess = Preprocessing(img)
    sharpened_img = preprocess.sharpen()
    if split:
        sharpened_img = preprocess.split_into_tiles(sharpened_img, preprocess_config['tile_size'])
    ######## END PREPROCESSING ########
    
    ######## BEGIN THRESHOLDING ########
    threshold_config = config['thresholding']
    
    threshold = Thresholding(sharpened_img, split, threshold_config)
    th, thresholded_img = threshold.apply()
    ######## END THRESHOLDING ########
    
    
    ######## PUT BOUNDING BOXES ########
    
    
    ######## BEGIN VISUALIZATION ########
    visualizer = Visualizer(thresholded_img) #visualize the image with napari using its numpy array
    #currently opening the napari visualizer stops the execution of the code
    ######## END VISUALIZATION ########
if __name__ == "__main__":
    main()
