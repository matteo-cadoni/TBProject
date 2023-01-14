import cv2 as cv
import numpy as np

class Thresholding:
    def __init__(self, img, config):
        self.img = img
        self.config = config
        print("Thresholding...")
        
    def check_algorithm(self):
        if self.config['algorithm'] == 'otsu' and not self.config['split']:
            return self.otsu_thresholding()
        
        if self.config['algorithm'] == 'otsu' and self.config['split']:
            tiles = self.split_into_tiles(tile_size = self.config['tile_size'])
            tiles_thresholded = self.otsu_thresholding_split(tiles)
            return self.reconstruct_image(tiles_thresholded)
        
        if self.config['algorithm'] == 'hard':
            return self.hard_thresholding(self.config['threshold'])
        
        if self.config['algorithm'] == 'adaptive_mean':
            return self.adaptive_thresholding(self.config['block_size'], self.config['c'])    
        
        if self.config['algorithm'] == 'adaptive_gaussian':
            return self.adpt_g_thresholding(self.config['block_size'], self.config['c'])


    def otsu_thresholding(self, img):
        """
        Threshold and binarize an image using Otsu's method

        :param image: image you want to threshold
        :return: ret: the computed threshold value
                th: binary image (image with the threshold applied, pixels above threshold are white = 255, pixels below threshold are black= 0)
        """
        print("Applying Otsu's thresholding...")
        ret, thresholded_image = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        return thresholded_image


    def otsu_thresholding_split(self, tiles):
        """
        Perform otsu thresholding on tile_size x tile_size images
        tiles that have blurry bacilli are furthermore sharpened

        :param img: image to be tresholded
        :return:    list of thresholded tiles
        """
        print("Applying Otsu's thresholding to each tile...")
        self.max_value = self.img.max()
        
        thresholded_tiles=[]
        for t in tiles:
            if self.check_all_white_tile(t):
                t[t>0] = 0   # set all pixels to black
                thresholded_tiles.append(t)
            else:    
                th = self.otsu_thresholding(t)
                if np.sum(th == 0) < 215 and np.sum(th == 0) > 200:
                    new_t = cv.addWeighted(th, 4, cv.blur(th, (30, 30)), -4, 128)
                    th_new = self.otsu_thresholding(new_t)
                    thresholded_tiles.append(th_new)
                else:
                    thresholded_tiles.append(th)
        return thresholded_tiles
        

    def hard_thresholding(self, threshold : int):
        """
        Implement hard threshold (a threshold manually imputed). 
        Take everything above "threshold" to be white and everything below "threshold" to be black.

        :param image: image to be thresholded
        :param threshold: hard threshold to be implemented
        :return: ret: threshold value
                th: binary image (pixels above threshold are white = 255, pixels below threshold are black= 0)
        """
        print("Applying hard thresholding...")
        ret,thresholded_image = cv.threshold(self.img, threshold, 255, cv.THRESH_BINARY)
        return ret, thresholded_image

    def adaptive_thresholding(self, block_size : int, c : int):
        """
        Apply adaptive thresholding to the image

        :param image: image to be thresholded
        :param block_size: size of the block used to compute the threshold
        :param c: constant subtracted from the mean or weighted mean
        :return: th: binary image (pixels above threshold are white = 255, pixels below threshold are black= 0)
        """
        print("Applying adaptive thresholding...")
        thresholded_image = cv.adaptiveThreshold(self.img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, c)
        return thresholded_image


    # compute adaptive thresholding
    def adpt_g_thresholding(self, block_size : int, c : int):
        """
        Threshold and binarize an image using adaptive thresholding using a Gaussian weighted sum

        :param image: image you want to threshold
        :return: ret: threshold value
                th: binary image
        """
        print("Applying adaptive gaussian thresholding...")
        # The threshold value is a gaussian-weighted sum of the neighbourhood (here of size 25) values minus the constant C (which is set to -7)
        thresholded_image = cv.adaptiveThreshold(self.img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, block_size, c)
        
        # ret is the computed value of the threshold AND th is the image with the threshold applied
        return thresholded_image
    
    #------------------------- SPLITTING AND RECONSTRUCTING IMAGE -------------------------#

    def split_into_tiles(self, tile_size = 16):
        """
        split image into tiles of shape tile_size * tile_size

        :param image: image to be split
        :param tile_size: dimensions of single tiles
        :return: tiles: list with the different tiles
        """
        print("Splitting image into tiles...")
        tiles = []
        for i in range(0, self.img.shape[0], tile_size):
            for j in range(0, self.img.shape[1], tile_size):
                tile = self.img[i:i+tile_size, j:j+tile_size]
                tiles.append(tile)
        return tiles
    
    def reconstruct_image(self, tiles: list):
        """
        :param tiles:    list with the different single tiles
        :param x_tiles:  how many tiles fit in the x axis
        :param y_tiles:  how many tiles fit in the y axis
        :return:         numpy array, reconstructed image
        """
        x_tiles = self.config['x_tiles']
        y_tiles = self.config['y_tiles']
        whole_img = np.zeros((x_tiles*tiles[0].shape[0], y_tiles*tiles[0].shape[1]))
        for i in range(x_tiles):
            for j in range(y_tiles):
                whole_img[i*tiles[0].shape[0]:(i+1)*tiles[0].shape[0], j*tiles[0].shape[1]:(j+1)*tiles[0].shape[1]] = tiles[i*y_tiles+j]
        return whole_img
    
    #------------------------- CHECKING IF TILE IS ALL WHITE -------------------------#
        
    def check_all_white_tile(self, t):
        """
        Check if we have a huge bright tile. if a tile_size x tile_size tile is all white,
        it is background and we set it black. Check based on global max pixel value

        :param img: tile to be checked if white
        :param max_value: max value pixel of whole image
        :return:
        """
        #print("Checking if current tile is all white...")
        
        if np.sum(t > 0.2 * self.max_value) > 0.8 * t.shape[0] * t.shape[1]:
            return True
        else:
            return False

    def set_zero(self, t):
        # set everything to black (0)
        t[t>0] = 0
        return t
        
    def apply(self):
        return self.check_algorithm()