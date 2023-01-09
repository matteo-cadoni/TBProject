import numpy as np
import cv2 as cv


class Postprocessing:
    
    def __init__(self, tiles, config):
        self.tiles = tiles
        self.config = config


    def cleaning_tiles(self):
        print("Cleaning tiles...")
        
        cleaned_tiles = []
        for t in self.tiles:
            # check if image is not a bacilli
            if self.check_image(t):  # the tile doesn't contain a bacilli
                t[t > 0] = 0
                cleaned_tiles.append(t)
            else: # the tile contains a bacilli
                cleaned_tiles.append(t)
        return cleaned_tiles

    def check_image(self, img: np.ndarray):
        """
        For every sub-image we check if its worth keeping or not
        number_of_black_pixels ---> maybe rely on scientific paper to find the optimal number

        :param img: image to be checked
        :return: bool
        """
        number_of_black_pixels = self.config['number_of_black_pixels']
        if np.sum(img == 0) > number_of_black_pixels:  #we have a bacilli
            return False
        else: #we have background
            return True
        
        
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
    
    def clean_connected_components(self, whole_tile):
        """
        Clean image with 2 approaches: delete connected components that have are up to 2 pixels
                                    connect bacilli that are separated by just one black pixel

        :param img: image to be cleaned
        :return:    cleaned image
        """
        # find connected components
        num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(np.uint8(whole_tile), connectivity=8)
        # stats = x,y,w,h,area

        # put to black connected components which area is equal to 1 or 2
        for i in range(1, num_labels):
            if stats[i][4] < 3:
                whole_tile[labels_im == i] = 0

        
        # connect the bacilli, by putting a white tile
        for i in range(1, whole_tile.shape[0]-1):
            for j in range(1, whole_tile.shape[1]-1):
                if whole_tile[i,j] == 0:
                    if (whole_tile[i-1,j] == 255 and whole_tile[i+1,j] == 255) or (whole_tile[i,j-1] == 255 and whole_tile[i,j+1] == 255) \
                            or (whole_tile[i-1,j-1] ==255 and whole_tile[i+1,j+1]) or (whole_tile[i-1,j+1] == 255 and whole_tile[i+1,j-1] == 255) \
                            or (whole_tile[i-1,j]== 255 and whole_tile[i+1,j+1]==255) or (whole_tile[i-1,j+1]==255 and whole_tile[i+1,j]==255)\
                            or (whole_tile[i-1,j]==255 and whole_tile[i+1,j-1]==255) or (whole_tile[i-1,j-1]==255 and whole_tile[i+1,j]==255)\
                            or (whole_tile[i,j-1]==255 and whole_tile[i+1,j+1]==255) or (whole_tile[i,j-1]==255 and whole_tile[i-1,j+1]==255)\
                            or (whole_tile[i,j+1]==255 and whole_tile[i+1,j-1]==255) or (whole_tile[i,j+1]==255 and whole_tile[i-1,j-1]==255):
                        whole_tile[i,j] = 255

        return whole_tile
    
    
            
    def apply(self):
        print("Applying postprocessing...")
        cleaned_tiles = self.cleaning_tiles()
        
        whole_img = self.reconstruct_image(cleaned_tiles)
        
        whole_img_cleaned = self.clean_connected_components(whole_img)
        return whole_img_cleaned