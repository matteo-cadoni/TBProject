from tkinter import *
from tkinter import messagebox


class InteractiveConfig:
    """
    Execute an interactive configuration window
    """
    def __init__(self):
        self.error = True
        self.window = Tk()
        # path of the smear
        self.path = StringVar()
        self.path.set('TB_sample/extern_Synlab_2156_17_3_MTB.czi')
        # tile to analyze (None for entire smear)
        self.tile = StringVar()
        # preprocessing algorithm
        self.preprocess_algorithm = StringVar()
        self.preprocess_algorithm.set('rescale')
        # thresholding algorithm
        self.algorithm = StringVar()
        self.algorithm.set('adaptive_gaussian')
        # hard threshold value
        self.hard_threshold_value = StringVar()
        self.hard_threshold_value.set('5000')
        # block size for adaptive thresholding
        self.block_size = StringVar()
        self.block_size.set('25')
        # c value for adaptive thresholding
        self.c = StringVar()
        self.c.set('-7')
        # use split otsu
        self.split_otsu = BooleanVar()
        self.split_otsu.set(False)
        # number of black pixels
        self.number_of_black_pixels = StringVar()
        self.number_of_black_pixels.set('215')
        # crop the image
        self.crop = BooleanVar()
        self.crop.set(False)
        # want to label dataset
        self.create_dataset = BooleanVar()
        self.create_dataset.set(False)
        # save dataset
        self.save_dataset = BooleanVar()
        self.save_dataset.set(False)
        # save stats
        self.save_stats = BooleanVar()
        self.save_stats.set(False)
        # do inference
        self.inference = BooleanVar()
        self.inference.set(False)
        # what kind of inference do you want to do
        self.inference_type = StringVar()
        self.inference_type.set('CNN')
        # visualize images
        self.show_images = BooleanVar()
        self.show_images.set(False)

    def run(self):
        """
        Run the interactive configuration window
        """
        self.window.title("Interactive Config")
        self.window.geometry('700x600')
        self.window.configure(background='white')
        # path input
        path_label = Label(self.window, text="Path of Smear")
        path_label.grid(column=0, row=0)
        path_entry = Entry(self.window, textvariable=self.path)
        path_entry.grid(column=1, row=0)
        # tile input
        tile_label = Label(self.window, text="Tile, number of the Tile we want to analyse, None for whole smear")
        tile_label.grid(column=0, row=1)
        tile_entry = Entry(self.window, textvariable=self.tile)
        tile_entry.grid(column=1, row=1)
        # preprocessing algorithm input
        preprocess_algorithm_label = Label(self.window, text="Preprocessing Algorithm, r"
                                                             "escale or sharp ")
        preprocess_algorithm_label.grid(column=0, row=2)
        preprocess_algorithm_entry = Entry(self.window, textvariable=self.preprocess_algorithm)
        preprocess_algorithm_entry.grid(column=1, row=2)
        # thresholding algorithm input
        algorithm_label = Label(self.window, text="Thresholding algorithm "
                                                  " (hard, otsu, adaptive_gaussian, adaptive_mean)")
        algorithm_label.grid(column=0, row=3)
        algorithm_entry = Entry(self.window, textvariable=self.algorithm)
        algorithm_entry.grid(column=1, row=3)
        # hard threshold value input
        hard_threshold_value_label = Label(self.window, text="Hard Threshold Value")
        hard_threshold_value_label.grid(column=0, row=4)
        hard_threshold_value_entry = Entry(self.window, textvariable=self.hard_threshold_value)
        hard_threshold_value_entry.grid(column=1, row=4)
        # block size input
        block_size_label = Label(self.window, text="Block Size for adaptive thresholding")
        block_size_label.grid(column=0, row=5)
        block_size_entry = Entry(self.window, textvariable=self.block_size)
        block_size_entry.grid(column=1, row=5)
        # c value input
        c_label = Label(self.window, text="C value for adaptive thresholding")
        c_label.grid(column=0, row=6)
        c_entry = Entry(self.window, textvariable=self.c)
        c_entry.grid(column=1, row=6)
        # split otsu input
        split_otsu_label = Label(self.window, text="Do you want to use split otsu?")
        split_otsu_label.grid(column=0, row=7)
        split_otsu_entry = Checkbutton(self.window, variable=self.split_otsu)
        split_otsu_entry.grid(column=1, row=7)
        # number of black pixels input
        number_of_black_pixels_label = Label(self.window, text="Number of Black Pixels used to clean the image")
        number_of_black_pixels_label.grid(column=0, row=8)
        number_of_black_pixels_entry = Entry(self.window, textvariable=self.number_of_black_pixels)
        number_of_black_pixels_entry.grid(column=1, row=8)
        # crop input
        crop_label = Label(self.window, text="Do you want to crop the image?")
        crop_label.grid(column=0, row=9)
        crop_entry = Checkbutton(self.window, variable=self.crop)
        crop_entry.grid(column=1, row=9)
        # create dataset input
        create_dataset_label = Label(self.window, text="Do you want to create a dataset?")
        create_dataset_label.grid(column=0, row=10)
        create_dataset_entry = Checkbutton(self.window, variable=self.create_dataset)
        create_dataset_entry.grid(column=1, row=10)
        # save dataset input
        save_dataset_label = Label(self.window, text="Do you want to save the dataset?")
        save_dataset_label.grid(column=0, row=11)
        save_dataset_entry = Checkbutton(self.window, variable=self.save_dataset)
        save_dataset_entry.grid(column=1, row=11)
        # save stats input
        save_stats_label = Label(self.window, text="Do you want to save the stats?")
        save_stats_label.grid(column=0, row=12)
        save_stats_entry = Checkbutton(self.window, variable=self.save_stats)
        save_stats_entry.grid(column=1, row=12)
        # inference input
        inference_label = Label(self.window, text="Do you want to do inference?")
        inference_label.grid(column=0, row=13)
        inference_entry = Checkbutton(self.window, variable=self.inference)
        inference_entry.grid(column=1, row=13)
        # inference type input
        inference_type_label = Label(self.window, text="What kind of inference do you want to do? (CNN, SVM, STATS)")
        inference_type_label.grid(column=0, row=14)
        inference_type_entry = Entry(self.window, textvariable=self.inference_type)
        inference_type_entry.grid(column=1, row=14)
        # show images input
        show_images_label = Label(self.window, text="Show Images")
        show_images_label.grid(column=0, row=15)
        show_images_entry = Checkbutton(self.window, variable=self.show_images)
        show_images_entry.grid(column=1, row=15)
        # finish button
        finish_button = Button(self.window, text="Finish", command=self.finish)
        finish_button.grid(column=0, row=16)
        # activate window
        self.window.mainloop()

        return [self.path, self.tile, self.preprocess_algorithm, self.algorithm,
                self.hard_threshold_value, self.block_size, self.c,
                self.split_otsu, self.number_of_black_pixels, self.crop,
                self.create_dataset, self.save_dataset, self.save_stats,
                self.inference, self.inference_type, self.show_images]

    def get_values(self):
        """
        Get the values of the input fields
        """
        # get path
        self.path = self.path.get()
        # if path is not a string or is empty, raise error
        if not isinstance(self.path, str) or self.path == "":
            raise ValueError("Path is not a string or is empty")

        # get tile
        tile_number = self.tile.get()
        if tile_number == 'None':
            self.tile = 'None'
        else:
            # if tile is not an integer, raise error
            try:
                self.tile = int(tile_number)
            # catch error of using int on a string
            except ValueError:
                raise ValueError("Tile is not an integer")

        # get preprocess algorithm
        self.preprocess_algorithm = self.preprocess_algorithm.get()
        if self.preprocess_algorithm not in ["rescale", "sharp"]:
            raise ValueError("Preprocessing algorithm is not rescale or sharp")

        # get algorithm
        self.algorithm = self.algorithm.get()
        if self.algorithm not in ["hard", "otsu", "adaptive_gaussian", "adaptive_mean"]:
            raise ValueError("Thresholding algorithm is not hard, otsu, adaptive_gaussian, or adaptive_mean")
        if self.preprocess_algorithm == "sharp" and self.algorithm != "otsu":
            raise ValueError("Wrong combination, sharp only works with otsu ")

        # get hard threshold value
        try:
            self.hard_threshold_value = int(self.hard_threshold_value.get())
        except ValueError:
            raise ValueError("Hard threshold value is not an integer")

        # get block size
        try:
            self.block_size = int(self.block_size.get())
        except ValueError:
            raise ValueError("Block size is not an integer")

        # get c
        try:
            self.c = int(self.c.get())
        except ValueError:
            raise ValueError("C value is not an integer")

        # get split otsu
        self.split_otsu = self.split_otsu.get()
        if self.algorithm != "otsu" and self.split_otsu:
            raise ValueError("Split only works with otsu")

        # get number of black pixels
        try:
            self.number_of_black_pixels = int(self.number_of_black_pixels.get())
        except ValueError:
            raise ValueError("Number of black pixels is not an integer")

        # get crop
        self.crop = self.crop.get()

        # get create dataset
        self.create_dataset = self.create_dataset.get()
        if (not self.crop) and self.create_dataset:
            raise ValueError("Can't create dataset without cropping")

        # get save dataset
        self.save_dataset = self.save_dataset.get()
        if (not self.crop) and (not self.create_dataset) and self.save_dataset:
            raise ValueError("Can't save dataset without cropping and creating dataset")

        # get save stats
        self.save_stats = self.save_stats.get()

        # get inference
        self.inference = self.inference.get()


        # get inference type
        self.inference_type = self.inference_type.get()
        if self.inference_type not in ["CNN", "SVM", "STATS"]:
            raise ValueError("Inference type is not CNN, SVM, or STATS")
        if self.inference and (self.inference_type == "CNN") and (not self.crop):
            raise ValueError("Can't do CNN inference without cropping")

        # get show images
        self.show_images = self.show_images.get()
        if self.tile == 'None' and self.show_images:
            raise ValueError("Can't show entire smear yet")

        # if there were no errors in getting the data det error to false
        self.error = False

    def finish(self):
        """
        Finish the input
        """
        try:
            self.get_values()
            self.window.destroy()
        # catch errors and show them in a messagebox, destroy window to start over
        except ValueError as e:
            messagebox.showerror("Please enter a valid value", str(e))
            self.window.destroy()


def change_yaml(config_values, config):
    """
    Function that changes the yaml file with the values from the interface

    param config_values: list of values from the interface
           config: yaml file to be changes
    return: Config: yaml file with the new values
    """
    config['load']['czi_path'] = config_values[0]
    config['load']['tile'] = config_values[1]
    config['preprocessing']['algorithm'] = config_values[2]
    config['thresholding']['algorithm'] = config_values[3]
    config['thresholding']['hard_threshold_param'] = config_values[4]
    config['thresholding']['block_size'] = config_values[5]
    config['thresholding']['c'] = config_values[6]
    config['thresholding']['split'] = config_values[7]
    config['postprocessing']['algorithm'] = config_values[3]
    config['postprocessing']['number_of_black_pixels'] = config_values[8]
    config['postprocessing']['crop'] = config_values[9]
    config['labelling_dataset']['create_dataset'] = config_values[10]
    config['saving']['save'] = config_values[11]
    config['saving']['save_stats'] = config_values[12]
    config['inference']['do_inference'] = config_values[13]
    config['inference']['prediction'] = config_values[14]
    config['visualization']['show'] = config_values[15]
    return config
