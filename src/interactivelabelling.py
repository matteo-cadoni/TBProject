from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image


def rescale_image(image):
    """ Rescale image to range 0-255

    parameters
    ----------
    image:
        image to be rescaled

    returns
    -------
    image:
        rescaled image
    """
    image = image - np.min(image)
    image = image / np.max(image)
    image = image * 255
    return image


def change_contrast(image, level):
    """ Change the contrast of an image.

    parameters
    ----------
    image:
        image to be changed
    level:
        level of contrast change

    returns
    -------
    image:
        image with changed contrast
    """
    # rescale image
    image = np.uint8(rescale_image(image))
    # create image object
    image = Image.fromarray(image)
    # define contrast level
    factor = (259 * (level + 255)) / (255 * (259 - level))

    # function that applies the contrast change
    def contrast(c):
        return 128 + factor * (c - 128)

    return image.point(contrast)


class InteractiveLabeling:
    """ Class that executes window for interactive labeling

    attributes
    ----------
    images
        list of images to be labeled
    labels
        list of labels, 1 for bacilli, 0 for not bacilli
    window
        tkinter window

    methods
    -------
    run()
        run the interactive labeling window
    bacilli_clicked()
        action for when the bacilli button is clicked
    not_bacilli_clicked()
        action for when the not bacilli button is clicked
    plot_next_image(i)
        plot the next image in the list
    """
    def __init__(self, images):
        """
        parameters
        ----------
        images
            list of images to be labeled
        """
        # initialize parameters
        self.images = images
        self.labels = np.array([])
        self.window = Tk()
        self.max_images = images.shape[0]

    def run(self):
        """ Run the interactive labeling window

        returns
        -------
        labels
            list of labels, 1 for bacilli, 0 for not bacilli
        """
        # create window
        self.window.title("Interactive Labeling")
        self.window.geometry('1200x1000')
        self.window.configure(background='white')
        # create buttons
        bacilli_button = Button(self.window, text="Bacilli")
        bacilli_button.grid(column=3, row=0)
        not_bacilli_button = Button(self.window, text="Not a Bacilli")
        not_bacilli_button.grid(column=3, row=1)
        # define button actions
        bacilli_button.configure(command=self.bacilli_clicked)
        not_bacilli_button.configure(command=self.not_bacilli_clicked)
        # create finish button
        finish_button = Button(self.window, text="Finish")
        finish_button.grid(column=3, row=2)
        finish_button.configure(command=self.window.destroy)
        # plot first image
        self.plot_next_image(0)
        # run window
        self.window.mainloop()
        return self.labels

    def bacilli_clicked(self):
        """ Action for when the bacilli button is clicked

        """
        # append label 1
        self.labels = np.append(self.labels, 1)
        # clear and close figure
        self.fig.clear()
        plt.close(self.fig)
        # check if all images have been labeled
        if self.labels.shape[0] == self.max_images:
            self.window.destroy()
        # plot next image
        self.plot_next_image(self.labels.shape[0])

    def not_bacilli_clicked(self):
        """ Action for when the not bacilli button is clicked

        """
        # append label 0
        self.labels = np.append(self.labels, 0)
        self.fig.clear()
        plt.close(self.fig)
        if self.labels.shape[0] == self.max_images:
            self.window.destroy()
        self.plot_next_image(self.labels.shape[0])

    def plot_next_image(self, i):
        """Plot the next image in the list

        parameters
        ----------
        i: index of image to be plotted
        """
        # initialize figure and axes
        self.fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(13, 10))
        # plot images
        ax1[0].imshow(self.images[i], cmap='gray')
        im2 = change_contrast(self.images[i], 100)
        ax2[0].imshow(im2, cmap='gray')
        im3 = change_contrast(self.images[i], 50)
        ax1[1].imshow(im3, cmap='gray')
        im4 = change_contrast(self.images[i], -100)
        ax2[1].imshow(im4, cmap='gray')
        # draw figure
        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas.get_tk_widget().grid(row=6, column=3)
