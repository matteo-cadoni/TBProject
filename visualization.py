import napari
import numpy as np
#VISUALIZATION
# visualize the image with napari using its numpy array
def visualize_napari(numpy_img: np.ndarray, name):
    """
    :param numpy_img: image to be visualized
    """
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(numpy_img, name=name)
    napari.run()


# visualize different images in the same moment
def visualize_all_list_napari(numpy_img_list: np.ndarray, names):
    """
    :param numpy_img_list: list containing different images to be visualized
    """
    with napari.gui_qt():
        viewer = napari.Viewer()
        for i, img in enumerate(numpy_img_list):
            viewer.add_image(img, name=names[i])
    napari.run()