
from aicsimageio.readers import CziReader
import h5py
import time
import os


class Loader:
    """
    Given a single sputum smear image made up of multiple tiles, load and transform it to numpy array
    """

    def __init__(self, czi_path, tile):
        """
        param czi_path: path to the czi file
                 tile: tile to be loaded or None
        """
        self.czi_path = czi_path
        self.tile = tile
        smear_folder = self.czi_path.split('/')[-1]
        smear_number = smear_folder.split('_')[2:5]
        # whole smear
        if self.tile == 'None':
            self.dataset_name = f'smear_{smear_number[0]}_{smear_number[1]}_{smear_number[2]}'
        # single tile
        else:
            self.dataset_name = f'tile_{tile}_smear_{smear_number[0]}_{smear_number[1]}_{smear_number[2]}'
        
    def read_array_from_h5(self, h5_path):
        """
        read the array from h5 file

        param h5_path: path to the h5 file
        """
        print(f"Reading array from {h5_path}...")
        h5file = h5py.File(h5_path, 'r')
        self.data_array = h5file[self.dataset_name][:]
        
    def save_array_to_h5(self, h5_path):
        """
        save the array to h5 file

        param h5_path: path to the h5 file
        """
        print(f"Saving array to {h5_path}...")
        h5file = h5py.File(h5_path, 'w')
        h5file.create_dataset(self.dataset_name, data=self.data_array)
                  
    def read_array_from_czi(self):
        """
        read the array from czi file
        """
        print(f"Reading array from {self.czi_path}...")
        reader = CziReader(self.czi_path)
        if self.tile == 'None':
            self.data_array = reader.get_image_data("MYX", C=0)
        else:
            self.data_array = reader.get_image_data("YX", M=self.tile, C=0)

    def load(self):
        """
        load the array from h5 file if it exists, otherwise read from czi file
        """
        print(f"Loading {self.dataset_name}...")
        # check if h5_file exists, otherwise create it
        h5_path = os.path.join('h5_data', self.dataset_name + '.h5')
        if os.path.isfile(h5_path):
            print("h5 file exists!")
            # check time to read h5 file
            start_time = time.time()
            self.read_array_from_h5(h5_path)
            end_time = time.time()
            print("Time to read h5 file: ", end_time - start_time)
        else:
            # check time to read czi file
            print("h5 file does not exist!")
            start_time = time.time()
            # create folder /h5_data if it does not exist
            if not os.path.exists('h5_data'):
                print("Folder h5_data not found, creating it")
                os.makedirs('h5_data')
            assert os.path.isfile(self.czi_path), "Please specify the path to the czi file"
            self.read_array_from_czi()
            end_time = time.time()
            print("Time to read czi file: ", end_time - start_time)
            self.save_array_to_h5(h5_path)
