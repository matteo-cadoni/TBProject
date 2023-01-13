from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class  InteractiveConfig:
    def __init__(self):

        self.window=Tk()
        self.path = StringVar()
        self.path.set('TB_sample/extern_Synlab_2156_17_3_MTB.czi')
        self.tile = StringVar()
        self.blur_deselection = BooleanVar()
        self.algorithm = StringVar()
        self.algorithm.set('otsu')
        self.hard_threshold_value = IntVar()
        self.hard_threshold_value.set(5000)
        self.number_of_black_pixels = IntVar()
        self.number_of_black_pixels.set(215)
        self.crop = BooleanVar()
        self.show_images = BooleanVar()
        self.save_images = BooleanVar()
        self.save_path = StringVar()
        self.save_path.set('extern_Synlab_2156_17_3_MTB.png')



    def run(self):
        self.window.title("Interactive Config")
        self.window.geometry('700x600')
        self.window.configure(background='white')

        path_label = Label(self.window, text="Path of Smear")
        path_label.grid(column=0, row=0)
        path_entry = Entry(self.window, textvariable=self.path)
        path_entry.grid(column=1, row=0)

        tile_label = Label(self.window, text="Tile, number of the Tile we want to analyse")
        tile_label.grid(column=0, row=1)
        tile_entry = Entry(self.window, textvariable=self.tile)
        tile_entry.grid(column=1, row=1)

        blur_deselection_label = Label(self.window, text="Do you want to deselect blurry images?")
        blur_deselection_label.grid(column=0, row=2)
        blur_deselection_entry = Checkbutton(self.window, variable=self.blur_deselection)
        blur_deselection_entry.grid(column=1, row=2)

        algorithm_label = Label(self.window, text="What algorithm do you want to use? (hard_threshold, otsu, adaptive)")
        algorithm_label.grid(column=0, row=3)
        algorithm_entry = Entry(self.window, textvariable=self.algorithm)
        algorithm_entry.grid(column=1, row=3)

        hard_threshold_value_label = Label(self.window, text="Hard Threshold Value")
        hard_threshold_value_label.grid(column=0, row=4)
        hard_threshold_value_entry = Entry(self.window, textvariable=self.hard_threshold_value)
        hard_threshold_value_entry.grid(column=1, row=4)

        number_of_black_pixels_label = Label(self.window, text="Number of Black Pixels used to clean the image")
        number_of_black_pixels_label.grid(column=0, row=5)
        number_of_black_pixels_entry = Entry(self.window, textvariable=self.number_of_black_pixels)
        number_of_black_pixels_entry.grid(column=1, row=5)

        crop_label = Label(self.window, text="Do you want to crop the image?, to get bacilli images")
        crop_label.grid(column=0, row=6)
        crop_entry = Checkbutton(self.window, variable=self.crop)
        crop_entry.grid(column=1, row=6)

        show_images_label = Label(self.window, text="Show Images")
        show_images_label.grid(column=0, row=7)
        show_images_entry = Checkbutton(self.window, variable=self.show_images)
        show_images_entry.grid(column=1, row=7)

        save_path_label = Label(self.window, text="Where do we save it?")
        save_path_label.grid(column=0, row=8)
        save_path_entry = Entry(self.window, textvariable=self.save_path)
        save_path_entry.grid(column=1, row=8)

        finish_button = Button(self.window, text="Finish", command=self.finish)
        finish_button.grid(column=0, row=9)

        #return entries from window
        self.window.mainloop()

        return [self.path, self.tile, self.blur_deselection, self.algorithm, self.hard_threshold_value,
                self.number_of_black_pixels, self.crop, self.show_images, self.save_path]
        # get the values from the entry boxes
    def get_values(self):
        self.path=self.path.get()
        self.tile=self.tile.get()
        self.blur_deselection=self.blur_deselection.get()
        self.algorithm=self.algorithm.get()
        self.hard_threshold_value=self.hard_threshold_value.get()
        self.number_of_black_pixels=self.number_of_black_pixels.get()
        self.crop=self.crop.get()
        self.show_images=self.show_images.get()
        self.save_path=self.save_path.get()

    def finish(self):
        self.get_values()
        self.window.destroy()


#function that changes yaml file
def change_yaml(config_values, config):
    config['load']['czi_path'] = config_values[0]
    config['load']['tile'] = config_values[1]
    config['load']['blur_deselection'] = config_values[2]
    config['thresholding']['algorithm'] = config_values[3]
    config['thresholding']['hard_threshold_value'] = config_values[4]
    config['postprocessing']['number_of_black_pixels'] = config_values[5]
    config['postprocessing']['crop'] = config_values[6]
    config['visualization']['show'] = config_values[7]
    config['visualization']['path'] = config_values[8]
    return config



