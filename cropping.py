import numpy as np
#given image and connected componets stats find center of mass of each connected component
def find_center_of_mass( stats):
    center_of_mass = []
    for i in range(1, stats.shape[0]):
        #find center of mass
        x = stats[i, 0]
        y = stats[i, 1]
        w = stats[i, 2]
        h = stats[i, 3]
        #find center of mass
        center_of_mass.append((x + w//2, y + h//2))
    return center_of_mass

#given image and center of mass list give back a list of images 50x50 pixels around the center of mass

def crop_images(image, center_of_mass):
    cropped_images = []
    for i in range(len(center_of_mass)):
        x = center_of_mass[i][0]
        y = center_of_mass[i][1]
        cropped_images.append(image[y-25:y+25, x-25:x+25])
    return cropped_images

#padding arrays to 50x50 pixels
def padd_images(image):
    padded_image=np.zeros((50,50))
    if image.shape[0] < 50 or image.shape[1] < 50:
        padded_image[0:image.shape[0], 0:image.shape[1]] = image
        return padded_image
    else:
        return image
