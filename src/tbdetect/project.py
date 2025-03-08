import yaml
import os

#from aicsimageio import AICSImage
#import napari
#from aicsimageio.readers import CziReader
#import cv2 as cv
#import logging  #maybe useful for debugging

from tbdetect.utils import *


def main():
    config_file = os.path.join(os.path.dirname(__file__), 'configs', 'thresholding.yaml')

    # read config from input.yaml
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Interactive config
    if config['load']['interactive_config']:
        config = interactive_config(config)
        
    # Load data
    load_config = config['load']
    img, loader = load(load_config)

    if load_config['tile'] == 'None':
        smear_pipeline(config, img, loader)
    else:
        tile_pipeline(config, img, loader)  
              
if __name__ == "__main__":
    main()