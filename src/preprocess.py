import numpy as np
import cv2 as cv


class Preprocessing:
    """
    either rescale or sharpen image
    """

    def __init__(self, img):
        self.img = img
        print("Preprocessing image...")

    def sharpen(self):
        """
         sharpen image using high-pass filter

        :return: sharp image
        """
        print("Sharpening image...")
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        return cv.filter2D(self.img, -1, kernel)

    def rescale(self):
        """
        rescale and convert image
        :return: rescaled image
        """
        rescaled_image = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img)) * 255
        # rescaled_image = np.round(rescaled_image).astype(np.uint16)
        rescaled_image = np.round(rescaled_image).astype('uint8')
        return rescaled_image
