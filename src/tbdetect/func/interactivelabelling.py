import sys
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class InteractiveLabeling():
    def __init__(self, images):
        self.images = images
        self.labels=np.array([])
        self.window=Tk()
        self.max_images=images.shape[0]

    def run(self):
        self.window.title("Interactive Labeling")
        self.window.iconbitmap('src/tbdetect/func/icon2.ico')
        self.window.geometry('800x800')
        self.window.configure(background='white')
        bacilli_button=Button(self.window, text="Bacilli")
        bacilli_button.grid(column=3, row=0, padx=5, pady=5)
        bacilli_button.configure(command=self.bacilli_clicked)
        not_bacilli_button=Button(self.window, text="Not a Bacilli")
        not_bacilli_button.grid(column=3, row=1, padx=5, pady=5)
        not_bacilli_button.configure(command=self.not_bacilli_clicked)
        # finish button
        finish_button=Button(self.window, text="Finish with this tile")
        finish_button.grid(column=3, row=2, padx=5, pady=5)
        finish_button.configure(command=self.window.destroy)
        # abort button
        abort_button = Button(self.window, text="Abort labelling")
        abort_button.grid(column=3, row=3, padx=5, pady=5)
        abort_button.configure(command=self.abort_clicked)

        self.plot_next_image(0)

        self.window.mainloop()

        return self.labels

    def bacilli_clicked(self):
        self.labels=np.append(self.labels, 1)
        self.figure.clear()
        plt.close(self.figure)
        if self.labels.shape[0] == self.max_images:
            self.window.destroy()
        self.plot_next_image(self.labels.shape[0])

    def not_bacilli_clicked(self):
        self.labels=np.append(self.labels, 0)
        self.figure.clear()
        plt.close(self.figure)
        if self.labels.shape[0] == self.max_images:
            self.window.destroy()
        self.plot_next_image(self.labels.shape[0])

    def plot_next_image(self,i):
        self.figure = plt.figure(figsize=(7, 6), dpi=100)
        plt.imshow(self.images[i], cmap='gray')
        canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        canvas.get_tk_widget().grid(row=6, column=3)

    def abort_clicked(self):
        sys.exit()