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
    
    def split_into_tiles(self, image, tile_size=32):
        """
        split image into tiles of shape tile_size*tile_size

        :param image: image to be split
        :param tile_size: dimensions of single tiles
        :return: tiles: list with the different tiles
        """
        print("Splitting image into tiles...")
        tiles = []
        for i in range(0, image.shape[0], tile_size):
            for j in range(0, image.shape[1], tile_size):
                tile = image[i:i+tile_size, j:j+tile_size]
                tiles.append(tile)
        return tiles
    
    # approach in which we sharp (instead of blur as the example) the image before applying the thresholding
    # sharpen the image using a high-pass filter TODO: can we do this better? sharp out better maybe in sub-images?
