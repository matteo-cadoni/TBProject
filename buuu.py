import cv2 as cv
import numpy as np
import napari
from aicsimageio.readers import CziReader

#add 5 pixels to the y values
def padd(img):
    img = np.pad(img, ((0, 1),(0,1)), 'constant')
    return img

def is_blurry(image):
    image= padd(image)
    values=np.array([])
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    #apply kernel to image and store output in list
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            #apply kernel to image
            value = np.sum(kernel*image[i-1:i+2,j-1:j+2])

    #get mean value of the list
    mean = np.mean(values)
    return mean


# get the mean value of the pixels in the image
def mean_value(image):

#PIPELINE (until the end)

# Load the image
reader = CziReader("TB_sample\extern_Synlab_2156_17_3_MTB.czi")
# Get whole image
smear = reader.get_image_data("MYX", C=0)
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(smear, name="smear")


mes=[]
for tile in smear:
    re=is_blurry(tile)
    mes.append(re)

#get mean of list
g=np.array(mes)

#sort numpy array
s=np.sort(g)
f=g.argsort()
print(f)
#apply same sort to smear
#smear_sorted=smear[g.argsort()]

print(g.max(), g.min(), g.mean(), g.std())
