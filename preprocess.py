import numpy as np
import cv2 as cv

class Preprocessing: 
    def __init__(self, img):
        self.img = img
        print("Preprocessing image...")

    #sharpen image using high-pass filter
    def sharpen(self: np.ndarray):
        """
        :param image: image to be sharpened
        :return: sharp image
        """
        print("Sharpening image...")
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        
        return cv.filter2D(self.img, -1, kernel)
