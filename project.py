import argparse
import yaml
import sys

from aicsimageio import AICSImage
import napari
from aicsimageio.readers import CziReader
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import logging as log

from visualization import Visualizer
from loader import Loader


def arguments_parser():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Tubercolosis Detection')
    parser.add_argument('config', type=str, default='configs/thresholding.yaml',
                        help='configure file for thresholding experiments')
    return parser

def main():
    #initialize logger
    #log.getLogger().addHandler(log.StreamHandler())
    #log.info('Starting program')
    
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
    
if __name__ == "__main__":
    main()
