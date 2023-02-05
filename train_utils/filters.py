import pandas as pd
import time
import tqdm
import numpy as np

class filter():
    
    def __init__(self, rmv_b, rmv_w) -> None:
        self.rmv_black_img = rmv_b
        self.rmv_white_img = rmv_w

    def remove_white_img(self, data):
        # given an input dataframe, check each image whether maximum value of pixel is 0
        data = data.reset_index(drop=True)
        n_removed_img = 0
        indexes = []
        for i, img in enumerate(tqdm.tqdm(data['image'], total=data.shape[0])):
            #if the number of black pixels is less then 250
            if np.sum(img[img==0]) < 750:
                n_removed_img += 1
                indexes.append(i)

        filtered_data = data.drop(indexes)
        filtered_data = filtered_data.reset_index(drop=True)

        print("Removed {} white images".format(n_removed_img))
        return filtered_data
    
    def remove_black_img(self, data):
        # given an input dataframe, check each image whether maximum value of pixel is 0
        data = data.reset_index(drop=True)
        n_removed_img = 0
        indexes = []
        for i, img in enumerate(tqdm.tqdm(data['image'], total=data.shape[0])):
            if img.max() == 0:
                n_removed_img += 1
                indexes.append(i)
                time.sleep(0.005)
                
        filtered_data = data.drop(indexes)
        filtered_data = filtered_data.reset_index(drop=True)

        print("Removed {} black images".format(n_removed_img))
        return filtered_data

    def apply(self, data):
        if self.rmv_black_img:
            print("Removing black images")
            data = self.remove_black_img(data)
        if self.rmv_white_img:
            print("Removing white images")
            data = self.remove_white_img(data)
        return data
        