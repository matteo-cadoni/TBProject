import cv2 as cv

class Thresholding:
    def __init__(self, img, split, config):
        self.img = img
        self.split = split
        self.config = config
        print("Thresholding image...")
        
    def check_algorithm_from_config(self):
        if self.config['algorithm'] == 'otsu':
            return self.otsu_thresholding()
        
        if self.config['algorithm'] == 'hard':
            return self.hard_thresholding(self.config['threshold'])
        
        if self.config['algorithm'] == 'adaptive':
            return 0, self.adaptive_thresholding(self.config['block_size'], self.config['c'])    
        
        if self.config['algorithm'] == 'gaussian':
            return 0, self.gaussian_thresholding(self.config['block_size'], self.config['c'])
    def otsu_thresholding(self):
        """
        Threshold and binarize an image using Otsu's method

        :param image: image you want to threshold
        :return: ret: the computed threshold value
                th: binary image (image with the threshold applied, pixels above threshold are white = 255, pixels below threshold are black= 0)
        """
        if self.split:
            print("Applying Otsu's thresholding to each tile...")
            thresholded_tiles_sharp=[]
            thresholded_tiles_treshold=[]
            for t in self.img:
                ret,th = cv.threshold(t, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
                thresholded_tiles_sharp.append(th)
                thresholded_tiles_treshold.append(ret)
            return thresholded_tiles_treshold, thresholded_tiles_sharp
        else:
            print("Applying Otsu's thresholding...")
            ret,th = cv.threshold(self.img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            return ret, th

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
        ret,th = cv.threshold(self.img, threshold, 255, cv.THRESH_BINARY)
        return ret, th

    def adaptive_thresholding(self, block_size : int, c : int):
        """
        Apply adaptive thresholding to the image

        :param image: image to be thresholded
        :param block_size: size of the block used to compute the threshold
        :param c: constant subtracted from the mean or weighted mean
        :return: th: binary image (pixels above threshold are white = 255, pixels below threshold are black= 0)
        """
        print("Applying adaptive thresholding...")
        th = cv.adaptiveThreshold(self.img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, c)
        return th

    def gaussian_thresholding(self, block_size : int, c : int):
        """
        Apply gaussian thresholding to the image

        :param image: image to be thresholded
        :param block_size: size of the block used to compute the threshold
        :param c: constant subtracted from the mean or weighted mean
        :return: th: binary image (pixels above threshold are white = 255, pixels below threshold are black= 0)
        """
        print("Applying gaussian thresholding...")
        th = cv.adaptiveThreshold(self.img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, c)
        return th
