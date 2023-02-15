import cv2 as cv
import numpy as np


def otsu_thresholding(img):
    """
    Threshold and binarize an image using Otsu's method

    :param img: image you want to threshold
    :return:  th: binary image (image with the threshold applied,
                  pixels above threshold are white = 255, pixels below threshold are black= 0)
    """
    print("Applying Otsu's thresholding...")
    ret, thresholded_image = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresholded_image


def set_zero(t):
    # set everything to black (0)
    t[t > 0] = 0
    return t


class Thresholding:
    """
    perform thresholding on each tile
    """

    def __init__(self, img, config):
        self.img = img
        self.config = config
        #print("Thresholding...")

    def check_algorithm(self):
        if self.config['algorithm'] == 'otsu' and not self.config['split']:
            return otsu_thresholding(self.img)

        if self.config['algorithm'] == 'otsu' and self.config['split']:
            tiles = self.split_into_tiles(tile_size=self.config['tile_size'])
            tiles_thresholded = self.otsu_thresholding_split(tiles)
            return self.reconstruct_image(tiles_thresholded)

        if self.config['algorithm'] == 'hard':
            return self.hard_thresholding(self.config['hard_threshold_param'])

        if self.config['algorithm'] == 'adaptive_mean':
            return self.adpt_m_thresholding(self.config['block_size'], self.config['c'])

        if self.config['algorithm'] == 'adaptive_gaussian':
            return self.adpt_g_thresholding(self.config['block_size'], self.config['c'])

    def otsu_thresholding_split(self, tiles):
        """
        Perform otsu thresholding on tile_size x tile_size images
        tiles that have blurry bacilli are furthermore sharpened

        :param tiles: images to be tresholded
        :return:    list of thresholded tiles
        """
        print("Applying Otsu's thresholding to each tile...")
        self.max_value = self.img.max()
        thresholded_tiles = []
        for t in tiles:
            if self.check_all_white_tile(t):
                # if tile is all white, set it to black
                t[t > 0] = 0
                thresholded_tiles.append(t)
            else:
                th = otsu_thresholding(t)
                if 215 > np.sum(th == 0) > 200:
                    new_t = cv.addWeighted(th, 4, cv.blur(th, (30, 30)), -4, 128)
                    th_new = otsu_thresholding(new_t)
                    thresholded_tiles.append(th_new)
                else:
                    thresholded_tiles.append(th)
        return thresholded_tiles

    def hard_thresholding(self, threshold: int):
        """
        Implement hard threshold (a threshold manually imputed). 
        Take everything above "threshold" to be white and everything below "threshold" to be black.

        :param threshold: hard threshold to be implemented
        :return: ret: threshold value
                th: binary image (pixels above threshold are white = 255, pixels below threshold are black= 0)
        """
        print("Applying hard thresholding...")
        ret, thresholded_image = cv.threshold(self.img, threshold, 255, cv.THRESH_BINARY)
        return thresholded_image

    def adpt_m_thresholding(self, block_size: int, c: int):
        """
        Apply adaptive thresholding to the image

        :param block_size: size of the block used to compute the threshold
        :param c: constant subtracted from the mean or weighted mean
        :return: thresholded_image: binary image (pixels above threshold are white = 255,
         pixels below threshold are black= 0)
        """
        print("Applying adaptive mean thresholding...")
        thresholded_image = cv.adaptiveThreshold(self.img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size,
                                                 c)
        return thresholded_image

    # compute adaptive thresholding
    def adpt_g_thresholding(self, block_size: int, c: int):
        """
        Threshold and binarize an image using adaptive thresholding using a Gaussian weighted sum

        :param block_size: parameter for adaptive thresholding
        :param c: parameter for adaptive thresholding
        :return: th: binary image
        """
        #print("Applying adaptive gaussian thresholding...")
        # The threshold value is a gaussian-weighted sum of the neighbourhood (here of size 25)
        # values minus the constant C (which is set to -7)
        thresholded_image = cv.adaptiveThreshold(self.img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                                 block_size, c)
        return thresholded_image

    # ------------------------- SPLITTING AND RECONSTRUCTING IMAGE -------------------------#

    def split_into_tiles(self, tile_size=16):
        """
        split image into tiles of shape tile_size * tile_size

        :param tile_size: dimensions of single tiles
        :return: tiles: list with the different tiles
        """
        print("Splitting image into tiles...")
        tiles = []
        for i in range(0, self.img.shape[0], tile_size):
            for j in range(0, self.img.shape[1], tile_size):
                tile = self.img[i:i + tile_size, j:j + tile_size]
                tiles.append(tile)
        return tiles

    def reconstruct_image(self, tiles: list):
        """
        :param tiles:    list with the different single tiles
        :return:         numpy array, reconstructed image
        """
        x_tiles = self.config['x_tiles']
        y_tiles = self.config['y_tiles']
        whole_img = np.zeros((x_tiles * tiles[0].shape[0], y_tiles * tiles[0].shape[1]))
        for i in range(x_tiles):
            for j in range(y_tiles):
                whole_img[i * tiles[0].shape[0]:(i + 1) * tiles[0].shape[0],
                j * tiles[0].shape[1]:(j + 1) * tiles[0].shape[1]] = tiles[i * y_tiles + j]
        return whole_img

    # ------------------------- CHECKING IF TILE IS ALL WHITE -------------------------#

    def check_all_white_tile(self, t):
        """
        Check if we have a huge bright tile. if a tile_size x tile_size tile is all white,
        it is background, and we set it black. Check based on global max pixel value

        :param t: tile to be checked
        :return:
        """
        # print("Checking if current tile is all white...")

        if np.sum(t > 0.2 * self.max_value) > 0.8 * t.shape[0] * t.shape[1]:
            return True
        else:
            return False

    def apply(self):
        return self.check_algorithm()
