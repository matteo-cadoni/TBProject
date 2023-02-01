import numpy as np
import cv2 as cv


class Preprocessing:
    """ Class tha preprocesses the image.
    Either rescale or sharpen image

    attributes
    ----------
    img : numpy.ndarray
        image to be preprocessed

    methods
    -------
    sharpen()
        sharpen image using high-pass filter
    rescale()
        rescale and convert image to uint8
    """

    def __init__(self, img):
        """
        parameters
        ----------
        img : numpy.ndarray
            image to be preprocessed

        """
        self.img = img
        print("Preprocessing image...")

    def sharpen(self):
        """ sharpen image using high-pass filter

        returns
        -------
        numpy.ndarray
            sharpened imag
        """
        print("Sharpening image...")
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        return cv.filter2D(self.img, -1, kernel)

    def rescale(self):
        """rescale and convert image to uint8

        returns
        -------
        rescaled_image : numpy.ndarray
            rescaled image
        """
        rescaled_image = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img)) * 255
        # rescaled_image = np.round(rescaled_image).astype(np.uint16)
        rescaled_image = np.round(rescaled_image).astype('uint8')
        return rescaled_image
