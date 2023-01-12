import cv2 as cv
import numpy as np
import napari
from aicsimageio.readers import CziReader

def is_blurry(image):
    #compute laplacian of image
    laplacian = cv.Laplacian(image, cv.CV_64F)
    #compute variance of laplacian
    measure=laplacian.var()
    #print(measure)
    return measure

#PIPELINE (until the end)

# Load the image
reader = CziReader("TB_sample\extern_Synlab_2156_17_3_MTB.czi")
# Get whole image
smear = reader.get_image_data("MYX", C=0)

#viewer = napari.view_image(smear, name='smear')
#viewer.add_image(smear)
#napari.run()

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
