from aicsimageio import AICSImage
from aicsimageio.readers import CziReader
import numpy as np
import h5py
import time
import os

class Loader():
# Given a single sputum smear image made up of multiple tiles, load and transform it to numpy array
    def __init__(self, czi_path, tile):
        self.czi_path = czi_path
        self.tile = tile
        if self.tile == 'None':
            self.dataset_name = 'smear'
        else:
            self.dataset_name = f'tile_{tile}'
        #self.reader = CziReader(self.path)
        #self.metadata = self.reader.metadata
        #self.data = self.reader.data
        
    def read_array_from_h5(self, h5_path):
        # read the array from h5 file
        print(f"Reading array from {h5_path}...")
        h5f = h5py.File(h5_path, 'r')
        self.data_array = h5f[self.dataset_name][:]
        
    def save_array_to_h5(self, h5_path):
        # save the array to h5 file
        print(f"Saving array to {h5_path}...")
        h5f = h5py.File(h5_path, 'w')
        h5f.create_dataset(self.dataset_name, data=self.data_array)
                  
    def read_array_from_czi(self):
        # read the array from czi file
        print(f"Reading array from {self.czi_path}...")
        reader = CziReader(self.czi_path)
        if self.tile == 'None':
            self.data_array = reader.get_image_data("MYX", C=0)
        else:
            self.data_array = reader.get_image_data("YX", M=self.tile, C=0)

        
    def load(self):
        #load the image
        print(f"###### BEGIN LOADING {self.dataset_name} ######")
        
        #check if h5_file exists, otherwise create it
        h5_path = os.path.join('h5_data', self.dataset_name + '.h5')

        if os.path.isfile(h5_path):
            print("h5 file exists!")
            
            #check time to read h5 file
            start_time = time.time()
            self.read_array_from_h5(h5_path)
            end_time = time.time()
            print("Time to read h5 file: ", end_time - start_time)
        else:
            #check time to read czi file
            print("h5 file does not exist!")
            start_time = time.time()
            
            assert os.path.isfile(self.czi_path), "Please specify the path to the czi file"
            
            self.read_array_from_czi()
            end_time = time.time()
            print("Time to read czi file: ", end_time - start_time)
            print("Saving array to h5 file")
            self.save_array_to_h5(h5_path)