import argparse
import yaml
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import napari

#from aicsimageio import AICSImage
#import napari
#from aicsimageio.readers import CziReader
#import cv2 as cv
#import logging  #maybe useful for debugging

from mixed_functions import *


def arguments_parser():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Tubercolosis Detection')
    parser.add_argument('config', type=str, default='configs/thresholding.yaml',
                        help='configure file for thresholding experiments')
    return parser



def main():
    parser = arguments_parser()
    pars_arg = parser.parse_args()
    
    # read config from input .yaml
    with open(pars_arg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Interactive config
    if config['load']['interactive_config']:
        config = interactive_config(config)
        
    # Load data
    load_config = config['load']
    img, loader = load(load_config)

    if load_config['tile'] == 'None':
        smear_pipeline(config, img)
    else:
        tile_pipeline(config, img, loader)  
              
if __name__ == "__main__":
    main()