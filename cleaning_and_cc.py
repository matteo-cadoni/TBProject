import numpy as np
from split_otsu import split_into_tiles
import cv2 as cv

def check_all_white_tile(img, max_value_global):
    """
    Check if we have a huge bright tile. if a 16 x 16 tile is all white--->
    we want it black. Check based on global max pixel value

    :param img: tile to be checked if white
    :param max_value: max value pixel of whole image
    :return:
    """
    if np.sum(img > 0.2 * max_value_global) > 0.9 * img.shape[0] * img.shape[1]:
        return True
    else:
        return False

def ignore_connected_component_list(image):
    #do we really need this?
    #sharpened_img = sharpen(img)
    #get the maximum of the sharpened img, needed to check ig image is all white
    max_value = image.max()
    tiles_sharpened = split_into_tiles(image, 16)
    list_indices_of_tiles_to_ignore=[]
    for i, tile1 in enumerate(tiles_sharpened):
        #check if mostly white #ATTENTION, im touching the og in memory. If yes set direclty to black
        if check_all_white_tile(tile1, max_value):
            list_indices_of_tiles_to_ignore.append(i)
            if i%94 !=0:
                list_indices_of_tiles_to_ignore.append(i-1)
            if i%95 !=0:
                list_indices_of_tiles_to_ignore.append(i-2)

            if i%93 !=0:
                list_indices_of_tiles_to_ignore.append(i+1)

            if i%92 !=0:
                list_indices_of_tiles_to_ignore.append(i+2)

            if i>94:
                list_indices_of_tiles_to_ignore.append(i-94)
            if i>188:
                list_indices_of_tiles_to_ignore.append(i-188)
            if i<11938:
                list_indices_of_tiles_to_ignore.append(i+94)
            if i<11844:
                list_indices_of_tiles_to_ignore.append(i+188)
            if i>94 and i%94 !=0:
                list_indices_of_tiles_to_ignore.append(i-95)
            if i>94 and i%(93) !=0:
                list_indices_of_tiles_to_ignore.append(i-93)
            if i<11938 and i%94 !=0:
                list_indices_of_tiles_to_ignore.append(i+93)
            if i<11938 and i%(93) !=0:
                list_indices_of_tiles_to_ignore.append(i+95)
            #maybe add more to diagonal

    return list_indices_of_tiles_to_ignore

def clean_connected_components(img: np.ndarray):
    """
    Clean image with 2 approaches: delete connected components that have are up to 2 pixels
                                   connect bacilli that are separated by just one black pixel

    :param img: image to be cleaned
    :return:    cleaned image
    """

    #find connected components
    num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(np.uint8(img), connectivity=8)
    #stats = x,y,w,h,area

    #put to black connected components which area is equal to 1 or 2
    for i in range(1, num_labels):
        if stats[i][4] < 3:
            img[labels_im == i] = 0
    #do not want to connect bacilli in original, want to connect after little components are gone
    img2 = img.copy()
    #connect the bacilli, by putting a white tile
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == 0:
                if (img[i - 1, j] == 255 and img[i + 1, j] == 255) or (img[i, j - 1] == 255 and img[i, j + 1] == 255) \
                        or (img[i - 1, j - 1] == 255 and img[i + 1, j + 1]) or (
                        img[i - 1, j + 1] == 255 and img[i + 1, j - 1] == 255) \
                        or (img[i - 1, j] == 255 and img[i + 1, j + 1] == 255) or (
                        img[i - 1, j + 1] == 255 and img[i + 1, j] == 255) \
                        or (img[i - 1, j] == 255 and img[i + 1, j - 1] == 255) or (
                        img[i - 1, j - 1] == 255 and img[i + 1, j] == 255) \
                        or (img[i, j - 1] == 255 and img[i + 1, j + 1] == 255) or (
                        img[i, j - 1] == 255 and img[i - 1, j + 1] == 255) \
                        or (img[i, j + 1] == 255 and img[i + 1, j - 1] == 255) or (
                        img[i, j + 1] == 255 and img[i - 1, j - 1] == 255):
                    img2[i, j] = 255

    return img2

#given tile number get coordinates in that tile
def get_coordinates_in_tile(tile_number):
    row= tile_number // 95
    col= tile_number % 95
    y=np.arange(row*16, row*16+16)
    x=np.arange(col*16, col*16+16)
    return x,y
#check if a coordinate is in the coordinates of a tile
def check_if_in_tile(x,y, tile_number):
    x_tile, y_tile = get_coordinates_in_tile(tile_number)
    if x in x_tile and y in y_tile:
        return True
    else:
        return False
#given e list on numbers, delete duplicates
def delete_duplicates(list):
    new_list=[]
    for item in list:
        if item not in new_list:
            new_list.append(item)
    return new_list



#given stats, remove from stats the connected components that are in the list of tiles to ignore
def remove_tiles_to_ignore(stats, list_tiles_to_ignore):
    print(stats)
    stats_cleaned=np.array([[0,0,0,0,0]])
    list_tiles_to_ignore=delete_duplicates(list_tiles_to_ignore)

    for i in range(1, len(stats)):
        if check_if_in_tiles_to_ignore(stats[i][0], stats[i][1], list_tiles_to_ignore):
            pass
        else:
            stats_cleaned=np.vstack((stats_cleaned,stats[i,:]))


    print(stats_cleaned)
    stats_cleaned=stats_cleaned[1:,:]
    return stats_cleaned

#check if coordinate is in any of the tiles to ignore
def check_if_in_tiles_to_ignore(x,y, list_tiles_to_ignore):
    if len(list_tiles_to_ignore)==0:
        return False
    else:
        for i in list_tiles_to_ignore:
            if check_if_in_tile(x,y,i):
                return True
            else:
                return False

