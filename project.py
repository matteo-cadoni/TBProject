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
from visualization import Visualizer

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
    final_image = postprocess.apply()
    ######## END POSTPROCESSING ########     
    
    
    ######## BEGIN VISUALIZATION ########
    visualizer = Visualizer(final_image)
    #currently opening the napari visualizer stops the execution of the code
    ######## END VISUALIZATION ########
    
if __name__ == "__main__":
    main()
