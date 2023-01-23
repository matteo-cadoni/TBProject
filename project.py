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

    # Interactive config
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
