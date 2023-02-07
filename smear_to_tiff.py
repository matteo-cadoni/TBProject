from aicsimageio.readers import CziReader
import matplotlib.pyplot as plt
import cv2 as cv

# read smear
reader = CziReader("TB_sample/extern_Synlab_2156_17_3_MTB.czi")

# load as numpy array
smear = reader.get_image_data("MYX", C=0)

# for adjusting the contrast
clahe = cv.createCLAHE(clipLimit=20.0, tileGridSize=(8,8))

# save all images of the smear in tiff format
for idx, img in enumerate(smear):
    # adjust contrast and save image
    # cl1 = clahe.apply(img)
    # plt.imsave(f'Smear_2156_17_3/tile_{idx}.tiff', cl1)
    # save image without adjusting contrast
    plt.imsave(f'Smear_2156_17_3/tile_{idx}.tiff', img)
    # save image with different colormap
    plt.imsave(f'Smear_2156_17_3/tile_{idx}.tiff', img, cmap='Blues')