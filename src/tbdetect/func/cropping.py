import cv2 as cv
import numpy as np

class Cropping:

    def __init__(self, original_tile, stats):
        self.original_tile = original_tile
        self.stats = stats

    def find_center_of_mass(self):
        center_of_mass = []
        for i in range(0, self.stats.shape[0]):
            # find center of mass
            x = self.stats[i, 0]
            y = self.stats[i, 1]
            w = self.stats[i, 2]
            h = self.stats[i, 3]
            # find center of mass
            center_of_mass.append((x + w // 2, y + h // 2))
        return center_of_mass

    # given image and center of mass list give back a list of images 50x50 pixels around the center of mass

    def crop_images(self, center_of_mass):
        cropped_images = []
        for i in range(len(center_of_mass)):
            x = center_of_mass[i][0]
            y = center_of_mass[i][1]
            cropped_images.append(self.original_tile[y - 25:y + 25, x - 25:x + 25])
        return cropped_images

    # padding arrays to 50x50 pixels
    def padd_images(self, image):
        padded_image = np.zeros((50, 50))
        if image.shape[0] < 50 or image.shape[1] < 50:
            padded_image[0:image.shape[0], 0:image.shape[1]] = image
            return padded_image
        else:
            return image

    def crop_and_pad(self):
        print("Cropping...")

        center_of_mass = self.find_center_of_mass()
        cropped_images = self.crop_images(center_of_mass)

        if len(cropped_images) > 1:
            a = np.array(cropped_images[0])
            a = self.padd_images(a)
            b = np.array(cropped_images[1])
            b = self.padd_images(b)
            cropped_numpy = np.stack((a, b), axis=0)

            for i, img in enumerate(cropped_images):
                if i > 1:
                    c = np.array(cropped_images[i])
                    c = self.padd_images(c)
                    cropped_numpy = np.concatenate((cropped_numpy, [c]), axis=0)
            return cropped_numpy
        elif len(cropped_images) == 1:
            a = np.array(cropped_images[0])
            a = self.padd_images(a)
            cropped_numpy = np.expand_dims(a, axis=0)
            return cropped_numpy
        else:
            h = "no images"
            return h

