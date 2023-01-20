from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image

class InteractiveLabeling():
    def __init__(self, images):

        self.images = images
        self.labels=np.array([])
        self.window=Tk()
        self.max_images=images.shape[0]

    def run(self):

        self.window.title("Interactive Labeling")
        self.window.geometry('1200x1000')
        self.window.configure(background='white')
        bacilli_button=Button(self.window, text="Bacilli")
        bacilli_button.grid(column=3, row=0)
        not_bacilli_button=Button(self.window, text="Not a Bacilli")
        not_bacilli_button.grid(column=3, row=1)

        bacilli_button.configure(command=self.bacilli_clicked)
        not_bacilli_button.configure(command=self.not_bacilli_clicked)

        finish_button=Button(self.window, text="Finish")
        finish_button.grid(column=3, row=2)
        finish_button.configure(command=self.window.destroy)

        self.plot_next_image(0)

        self.window.mainloop()
        return self.labels



    def bacilli_clicked(self):
        self.labels=np.append(self.labels, 1)
        self.fig.clear()
        plt.close(self.fig)
        if self.labels.shape[0] == self.max_images:
            self.window.destroy()
        self.plot_next_image(self.labels.shape[0])

    def not_bacilli_clicked(self):
        self.labels=np.append(self.labels, 0)
        self.fig.clear()
        plt.close(self.fig)
        if self.labels.shape[0] == self.max_images:
            self.window.destroy()
        self.plot_next_image(self.labels.shape[0])

    def plot_next_image(self,i):
        self.fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(13, 10))
        #self.figure = plt.figure(figsize=(7, 6), dpi=100)
        #plt.imshow(self.images[i], cmap='gray')
        ax1[0].imshow(self.images[i], cmap='gray')
        im2 = self.change_contrast(self.images[i], 100)
        ax2[0].imshow(im2, cmap='gray')
        im3 = self.change_contrast(self.images[i], 50)
        ax1[1].imshow(im3, cmap='gray')
        im4 = self.change_contrast(self.images[i], -100)
        ax2[1].imshow(im4, cmap='gray')
        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas.get_tk_widget().grid(row=6, column=3)

    def change_contrast(self, image, level):
        image = np.uint8(self.rescale_image(image))
        image = Image.fromarray(image)
        factor = (259 * (level + 255)) / (255 * (259 - level))
        def contrast(c):
            return 128 + factor * (c - 128)
        return image.point(contrast)

    def rescale_image(self, image):
        image = image - np.min(image)
        image = image / np.max(image)
        image = image * 255
        return image