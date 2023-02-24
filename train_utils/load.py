import pandas as pd
import os
from .filters import filter


class loader():   
    def __init__(self, config) -> None:
        self.all_data_path = config['all_data_path']
        self.data_path_list = config['data_path_list']
        self.sampling_percentage = config['sampling_percentage']
        self.remove_black_img = config['remove_black_img']
        self.remove_white_img = config['remove_white_img']

    def load(self):     
        '''
        Load data from .pkl file(s).
        if "all_data_path" path is given, reads it from there,
        otherwise concatenates all dataframes from "data_path_list" into one
        and saves it in "all"
        '''
        print("--------------------------------------")
        print('Loading data from pkl file')
        if os.path.exists(self.all_data_path):
            data = self.load_from_path()
        else:
            data = self.load_from_paths()        
            #save data to all_data_path
            self.save_to_path(data)       
        print('Data loaded, shape: ', data.shape)
        return data        
    def load_from_paths(self):
        print("Loading from input data paths: ", self.data_path_list)
        data = []
        for i, path in enumerate(self.data_path_list):
            data.append(pd.read_pickle(path))
            print("Added data from path and shape: ", path, data[i].shape)
        return pd.concat(data)
    
    def save_to_path(self, data):
        print("Saving data to path: ", self.all_data_path)
        data.to_pickle(self.all_data_path)
    
    def load_from_path(self):
        print("Found all data at path: ", self.all_data_path)
        return pd.read_pickle(self.all_data_path)
    
    def apply_filters(self, data):
        print("\nApplying filters: ")
        f = filter(self.remove_black_img, self.remove_white_img)
        
        # check if data is already saved as data_filtered.pth
        if os.path.exists("labelled_data/data_filtered.pkl") is False:
            data = f.apply(data)
            # save data to data_filtered.pth
            # print("Saving data to path: ", "labelled_data/data_filtered.pkl")
            # data.to_pickle("labelled_data/data_filtered.pkl")
            return data          
# randomly sampled subset of input dataframe with sampling probability p
def sample(df, p, random_state=42):
    df_sampled = df.sample(frac=p, random_state=random_state)
    return df_sampled
        
    