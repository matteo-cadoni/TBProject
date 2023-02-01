""" Automated Detection of Tuberculosis Bacilli in Whole-Slide Images

This script allows to process whole-slide images of sputum smear samples
and detect the number of bacilli in each image.

The script processes whole-slide images of the format ".czi". Software
requirements are listed in the requirements.txt file.

Different results can be obtained by changing the configuration file.
Refer to this file for more information.

The script can be run from the command line as follows:
   python project.py configs/thresholding.yaml.
To run the script after module installation, use:
   project.run()
"""


import argparse
import yaml
from src.smear_function import *
from src.tile_function import *


def arguments_parser():
    """
    Parse arguments from config file
    """

    parser = argparse.ArgumentParser('Tuberculosis Detection')
    parser.add_argument('config', type=str, default='configs/thresholding.yaml',
                        help='configure file for thresholding experiments')
    return parser


def main():
    # Initialize arguments
    parser = arguments_parser()
    pars_arg = parser.parse_args()
    
    # read config from input .yaml
    with open(pars_arg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start Interactive configuration
    if config['load']['interactive_config']:
        config = interactive_config(config)
        
    # Load data
    load_config = config['load']
    img, loader = load(load_config)

    # Execute functions
    if load_config['tile'] == 'None':
        smear_pipeline(config, img, loader)
    else:
        tile_pipeline(config, img, loader)  


if __name__ == "__main__":
    main()
