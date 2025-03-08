from tkinter import *
from tkinter import messagebox


class  InteractiveConfig:
    def __init__(self):

        self.error = True

        self.window=Tk()

        #path of the smear
        self.path = StringVar()
        self.path.set('TB_sample/extern_Synlab_2156_17_3_MTB.czi')

        #tile to analyze give it a string, later analyze if its a None or tile
        self.tile = StringVar()

        #thresholding algorithm
        self.algorithm = StringVar()
        self.algorithm.set('adaptive_gaussian')
 
        #want to label dataset
        self.create_dataset = BooleanVar()
        self.create_dataset.set(False)

        #save dataset
        self.save_dataset = BooleanVar()
        self.save_dataset.set(False)
  
        #do inference
        self.inference = BooleanVar()
        self.inference.set(False)

        # what kind of inference do you want to do
        self.inference_type = StringVar()
        self.inference_type.set('')

        #visualize images
        self.show_images = BooleanVar()
        self.show_images.set(True)


    def run(self):
        self.window.title("Interactive Configuration")
        self.window.iconbitmap('src/tbdetect/func/icon3.ico')
        self.window.geometry('600x650')
        self.window.configure(background='white')

        # path input
        path_label = Label(self.window, text="Please enter the path of the smear you want to analyse \n (e.g. TB_sample/extern_Synlab_2156_17_3_MTB.czi)")
        path_label.grid(column=0, row=0, padx=5, pady=5)
        path_entry = Entry(self.window, textvariable=self.path)
        path_entry.grid(column=1, row=0, padx=5, pady=5)
        # tile input
        tile_label = Label(self.window, text="Please enter the number of the tile you want to analyse, \n write None for analysing the whole smear.")
        tile_label.grid(column=0, row=1, padx=5, pady=5)
        tile_entry = Entry(self.window, textvariable=self.tile)
        tile_entry.grid(column=1, row=1, padx=5, pady=5)
        # thresholding algorithm input
        algorithm_label = Label(self.window, text="Please select a thresholding algorithm: \n hard, otsu, adaptive_gaussian, adaptive_mean")
        algorithm_label.grid(column=0, row=2, padx=5, pady=5)
        algorithm_entry = Entry(self.window, textvariable=self.algorithm)
        algorithm_entry.grid(column=1, row=2, padx=5, pady=5)
        # create dataset input
        create_dataset_label = Label(self.window, text="Would you like to label the objects in the image \n with the interactive labeling tool?")
        create_dataset_label.grid(column=0, row=3, padx=5, pady=5)
        create_dataset_entry = Checkbutton(self.window, variable=self.create_dataset)
        create_dataset_entry.grid(column=1, row=3, padx=5, pady=5)
        # save dataset input
        save_dataset_label = Label(self.window, text="Would you like to save the dataset you just labelled?")
        save_dataset_label.grid(column=0, row=4, padx=5, pady=5)
        save_dataset_entry = Checkbutton(self.window, variable=self.save_dataset)
        save_dataset_entry.grid(column=1, row=4, padx=5, pady=5)
        # inference input
        inference_label = Label(self.window, text="Would you like to do inference?")
        inference_label.grid(column=0, row=5, padx=5, pady=5)
        inference_entry = Checkbutton(self.window, variable=self.inference)
        inference_entry.grid(column=1, row=5, padx=5, pady=5)
        # inference type input
        inference_type_label = Label(self.window, text="What kind of inference would you like to do? \n (CNN, SVM, STATS)")
        inference_type_label.grid(column=0, row=6, padx=5, pady=5)
        inference_type_entry = Entry(self.window, textvariable=self.inference_type)
        inference_type_entry.grid(column=1, row=6, padx=5, pady=5)
        # show images input
        show_images_label = Label(self.window, text="Show Images")
        show_images_label.grid(column=0, row=7, padx=5, pady=5)
        show_images_entry = Checkbutton(self.window, variable=self.show_images)
        show_images_entry.grid(column=1, row=7, padx=5, pady=5)
        # finish button
        finish_button = Button(self.window, text="Finish", command=self.finish)
        finish_button.grid(column=0, row=8, padx=5, pady=5)
        # abort button
        abort_button = Button(self.window, text="Abort", command=self.abort)
        abort_button.grid(column=0, row=9, padx=5, pady=5)

        self.window.mainloop()

        # return the values from the entry boxes
        return [self.path, self.tile, self.algorithm,
                self.create_dataset, self.save_dataset, self.inference, 
                self.inference_type, self.show_images]
    
    # get the values from the entry boxes
    def get_values(self):
        self.path = self.path.get()
        # if path is not a string or is empty, raise error
        if not isinstance(self.path, str) or self.path == "":
            raise ValueError("Path is not a string or is empty")
        
        tile_number = self.tile.get()
        if tile_number == 'None':
            self.tile = 'None'
        else:
            #if tile is not an integer, raise error
            try:
                self.tile = int(tile_number)
            #catch error of using int on a string
            except ValueError:
                raise ValueError("Tile is not an integer")
            
        self.algorithm = self.algorithm.get()
        if self.algorithm not in ["hard", "otsu", "adaptive_gaussian", "adaptive_mean"]:
            raise ValueError("Thresholding algorithm is not hard, otsu, \n adaptive_gaussian or adaptive_mean, please check spelling.")
        
        self.create_dataset = self.create_dataset.get()
        self.save_dataset = self.save_dataset.get()
        if (not self.create_dataset) and self.save_dataset:
            raise ValueError("Can't save a dataset without creating a dataset")
        
        self.inference = self.inference.get()
        # get inference type
        self.inference_type = self.inference_type.get()
        if self.inference_type not in ["CNN", "SVM", "STATS", ""]:
            raise ValueError("Inference type is not CNN, SVM, or STATS")
        
        self.error = False

    def finish(self):
        try:
            self.get_values()
            self.window.destroy()

        except ValueError as e:
            messagebox.showerror(e, "Please enter a valid value")
            self.window.destroy()

    def abort(self):
        self.window.destroy()
        self.error = False

#function that changes yaml file
def change_yaml(config_values, config):
    config['load']['czi_path'] = config_values[0]
    config['load']['tile'] = config_values[1]
    config['thresholding']['algorithm'] = config_values[2]
    config['postprocessing']['algorithm'] = config_values[2]
    config['labelling_dataset']['create_dataset'] = config_values[3]
    config['saving']['save'] = config_values[4]
    config['inference']['do_inference'] = config_values[5]
    config['inference']['prediction'] = config_values[6]
    config['visualization']['show'] = config_values[7]
    return config