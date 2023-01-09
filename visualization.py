import napari
import numpy as np
import matplotlib.pyplot as plt

def visualize_all_list_napari(numpy_img_list: np.ndarray,names):
    """
    :param numpy_img_list: list containing different images to be visualized
    """
    with napari.gui_qt():
        viewer = napari.Viewer()
        for i, img in enumerate(numpy_img_list):
            viewer.add_image(img, name=names[i])
