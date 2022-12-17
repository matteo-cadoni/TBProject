import napari
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, img, name='napari'):
        self.img = img
        with napari.gui_qt():
            self.viewer = napari.Viewer()
            self.viewer.add_image(img, name=name)
            napari.run()
    # visualize the image with napari using its numpy array
    
    def add_image_napari(self, img):
        self.viewer.add_image(img)
    
    def visualize_napari(self, img):
        """
        :param numpy_img: image to be visualized
        """
        self.viewer.add_image(img)

    # visualize different images in the same moment, a list of images is passed as argument and we visualize them all
    def visualize_all_list_napari(self, numpy_img_list: np.ndarray):
        """
        :param numpy_img_list: list containing different images to be visualized
        """
        with napari.gui_qt():
            viewer = napari.Viewer()
            for img in numpy_img_list:
                viewer.add_image(img)
    
    # plot histogram of pixel intensity
    def plot_histogram(self):
        plt.hist(self.img.ravel(), self.img.max(), [0, self.img.max()])
        plt.show()