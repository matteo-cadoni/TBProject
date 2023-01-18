from src.loader import Loader
from src.preprocess import Preprocessing
from src.interactive_config import InteractiveConfig, change_yaml
from src.visualization import visualize_all_list_napari, add_bounding_boxes, is_blurry_laplacian

def interactive_config(config):  
    while True:
        i_c = InteractiveConfig()
        configurations = i_c.run()
        if i_c.error:
            continue
        else:
            break
    return change_yaml(configurations, config)

def load(load_config):
    loader = Loader(load_config['czi_path'], load_config['tile'])
    loader.load()
    img = loader.data_array
    if load_config['tile'] == 'None':
        print(f"Smear succesfully loaded, shape: {img.shape}")
    else: 
        print(f"Tile succesfully loaded, shape: {img.shape}")
    return img, loader

def preprocess(preprocess_config, tile):
    preprocess = Preprocessing(tile)
    if preprocess_config['algorithm'] == "sharp":
        return preprocess.sharpen()
    if preprocess_config['algorithm'] == "rescale":
        return preprocess.rescale()