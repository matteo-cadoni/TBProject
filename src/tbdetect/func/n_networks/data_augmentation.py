import pandas as pd
import numpy as np

class DataAug:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def augment(self):
        new_dataframe = pd.DataFrame()
        for i in range(0, self.dataframe.shape[0]):
            img = self.dataframe.iloc[i]['image']
            # rotate image by 90 degrees with numpy
            img1 = np.rot90(img)
            # rotate image by 180 degrees with numpy
            img2 = np.rot90(img1)
            # rotate image by 270 degrees with numpy
            img3 = np.rot90(img2)



            d1 = {'image': img, 'label': self.dataframe.iloc[i]['label']}
            d2 = {'image': img1, 'label': self.dataframe.iloc[i]['label']}
            d3 = {'image': img2, 'label': self.dataframe.iloc[i]['label']}
            d4 = {'image': img3, 'label': self.dataframe.iloc[i]['label']}

            #use pandas concat
            new_dataframe = pd.concat([new_dataframe, pd.DataFrame([d1, d2, d3, d4])])

        return new_dataframe
