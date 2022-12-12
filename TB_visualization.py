import numpy as np
import aicspylibczi
from aicsimageio.readers import CziReader
import napari

reader = CziReader("extern_Synlab_2156_17_3_MTB.czi")

# img = reader.
img = reader.get_image_dask_data("MYX", C=0)

# create the viewer and display the image
viewer = napari.view_image(img)

# start the event loop and show the viewer
napari.run()
