import numpy as np
from interactivelabeling import InteractiveLabeling

images=np.load('cropped_images_real_674.npy')
il=InteractiveLabeling(images)
labels=il.run()
print(labels)