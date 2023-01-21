import yaml
import os

#from aicsimageio import AICSImage
#import napari
#from aicsimageio.readers import CziReader
#import cv2 as cv
#import logging  #maybe useful for debugging

from tbdetect.utils import *

"""
def arguments_parser():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Tubercolosis Detection')
    parser.add_argument('config', type=str, default='tbdetect/configs/thresholding.yaml',
                        help='configure file for thresholding experiments')
    return parser
    """

def main():
    """
    parser = arguments_parser()
    pars_arg = parser.parse_args()
    """
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
        smear_pipeline(config, img)
    else:
        tile_pipeline(config, img, loader)  
              
if __name__ == "__main__":
    main()