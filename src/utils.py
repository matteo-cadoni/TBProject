from src.loader import Loader
from src.preprocess import Preprocessing
from src.interactive_config import InteractiveConfig, change_yaml
import numpy as np


def interactive_config(config):
    """
    Interactive config for the user to change the config file.
    Initialization function for the InteractiveConfig class.

    :param config: config file
    """
    while True:
        i_c = InteractiveConfig()
        configurations = i_c.run()
        if i_c.error:
            continue
        else:
            break
    return change_yaml(configurations, config)


def load(load_config):
    """
    Load the data from the config file.
    Initialization function for the Loader class.

    :param load_config: config file
    """
    loader = Loader(load_config['czi_path'], load_config['tile'])
    loader.load()
    img = loader.data_array
    if load_config['tile'] == 'None':
        print(f"Smear succesfully loaded, shape: {img.shape}")
    else: 
        print(f"Tile succesfully loaded, shape: {img.shape}")
    return img, loader


def preprocess(preprocess_config, tile):
    """
    Preprocess the data from the config file.
    Initialization function for the Preprocessing class.
    """
    preprocessing = Preprocessing(tile)
    if preprocess_config['algorithm'] == "sharp":
        return preprocessing.sharpen()
    if preprocess_config['algorithm'] == "rescale":
        return preprocessing.rescale()


def clean_stats(stats):
    """
    Delete connected components that are too small, and
    connected components that are too large.
    """
    # make a copy of stats
    stats1 = stats.copy()
    # delete
    for i in range(0, stats.shape[0]):
        if stats[i, 4] > 625:
            stats1 = np.delete(stats, i, 0)
        if stats[i, 4] < 25:
            stats1 = np.delete(stats, i, 0)
    return stats1
