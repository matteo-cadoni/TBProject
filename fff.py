#use sobel operator to detect edges

import cv2
import numpy as np

img = cv2.imread('test.jpg',0)
edges = cv2.Canny(img,100,200)

