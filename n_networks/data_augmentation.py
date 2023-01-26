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
            # flip image horizontally with numpy
            img4 = np.fliplr(img)
            # flip image vertically with numpy
            img5 = np.flipud(img)


            d1 = {'image': img, 'label': self.dataframe.iloc[i]['label']}
            d2 = {'image': img1, 'label': self.dataframe.iloc[i]['label']}
            d3 = {'image': img2, 'label': self.dataframe.iloc[i]['label']}
            d4 = {'image': img3, 'label': self.dataframe.iloc[i]['label']}
            d5 = {'image': img4, 'label': self.dataframe.iloc[i]['label']}
            d6 = {'image': img5, 'label': self.dataframe.iloc[i]['label']}

            #use pandas concat
            new_dataframe = pd.concat([new_dataframe, pd.DataFrame([d1, d2, d3, d4, d5, d6])])

        return new_dataframe
