import numpy as np
import cv2 as cv

#sharpening methods

#classic sharpening
def sharpen(image: np.ndarray):
    """
    Sharpen the image
    :param image: image to be sharpened
    :return: sharp image
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv.filter2D(image, -1, kernel)

#more aggressive sharpening, too aggressive
def sharpen2(img):
    return cv.addWeighted(img, 4, cv.blur(img, (30, 30)), -4, 128)


#thresholding methods

#otsu thresholding
def otsu_thresholding(image: np.ndarray):
    """
    Threshold and binarize an image using Otsu's method

    :param image: image you want to threshold
    :return:  th_img: binary image
    """
    ret, th_img = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return th_img

#split and reconsruction

# split the whole images into tiles
def split_into_tiles(image: np.ndarray, tile_size: int):
    """
    split image into tiles of shape tile_size*tile_size

    :param image: image to be split
    :param tile_size: dimensions of single tiles
    :return: tiles: list with the different tiles
    """
    tiles = []
    for i in range(0, image.shape[0], tile_size):
        for j in range(0, image.shape[1], tile_size):
            tile = image[i:i + tile_size, j:j + tile_size]
            tiles.append(tile)
    return tiles

#reconstruct image from different tiles given the number of tiles
# in x and y direction and a list of tiles
def reconstruct_image(tiles: list, x_tiles: int, y_tiles: int):
    """
    :param tiles:    list with the different single tiles
    :param x_tiles:  how many tiles fit in the x axis
    :param y_tiles:  how many tiles fit in the y axis
    :return:         numpy array, reconstructed image
    """
    big_image = np.zeros((x_tiles * tiles[0].shape[0], y_tiles * tiles[0].shape[1]))
    for i in range(x_tiles):
        for j in range(y_tiles):
            big_image[i * tiles[0].shape[0]:(i + 1) * tiles[0].shape[0],
            j * tiles[0].shape[1]:(j + 1) * tiles[0].shape[1]] = tiles[i * y_tiles + j]
    return big_image

#otsu cleaned split thresholding
def otsu_cleaned_split_thresholding(img):
    """
    Performe otsu thresholding on 16 x 16 images, then clean the image,
     delete the noise

    :param img: image to be tresholded
    :return:    tresholded clean image
    """
    # sharpen image
    sharpened_img = sharpen(img)
    tiles_sharpened = split_into_tiles(sharpened_img,16)
    thresholded_tiles = []
    for t in tiles_sharpened:

        th = otsu_thresholding(t)
        if np.sum(th == 0) < 215 and np.sum(th == 0) > 200:
            new_t = sharpen2(t)
            th_new = otsu_thresholding(new_t)
            thresholded_tiles.append(th_new)
        else:
            thresholded_tiles.append(th)

    #list with the thresholded tiles size 16x16

    #clean
    cleaned_tiles = []
    for tl in thresholded_tiles:
        #check if image is not a bacilli
        if check_image(tl):
            #im not a bacilli
            m = set_zero(tl)
            cleaned_tiles.append(m)
        else:
            #i am a bacilli
            cleaned_tiles.append(tl)

    #reconstruct
    reconstructed_clean_image = reconstruct_image(cleaned_tiles, 128, 94)

    return reconstructed_clean_image

#check if we want the tile or not
def check_image(img: np.ndarray):
    """
    For every sub-image we check if its worth keeping or not
    215 pretty hard-coded---> maybe rely on scientific paper to find the optimal number

    :param img: image to be checked
    :return: bool
    """

    if np.sum(img == 0) > 215:  #we have a bacilli
        return False
    else:
        return True

#set image to zero
def set_zero(img):
    h = img
    h[h > 0] = 0
    return h
