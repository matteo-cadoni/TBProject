from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class InteractiveLabeling():
    def __init__(self, images):

        self.images = images
        self.labels=np.array([])
        self.window=Tk()

    def run(self):

        self.window.title("Interactive Labeling")
        self.window.geometry('700x600')
        self.window.configure(background='white')
        bacilli_button=Button(self.window, text="Bacilli")
        bacilli_button.grid(column=3, row=0)
        not_bacilli_button=Button(self.window, text="Not a Bacilli")
        not_bacilli_button.grid(column=3, row=1)

        bacilli_button.configure(command=self.bacilli_clicked)
        not_bacilli_button.configure(command=self.not_bacilli_clicked)
        self.plot_next_image(0)

        self.window.mainloop()
        return self.labels

        pass

    def bacilli_clicked(self):
        self.labels=np.append(self.labels, 1)
        self.figure.clear()
        plt.close(self.figure)
        self.plot_next_image(self.labels.shape[0])

    def not_bacilli_clicked(self):
        self.labels=np.append(self.labels, 0)
        self.figure.clear()
        plt.close(self.figure)
        self.plot_next_image(self.labels.shape[0])

    def plot_next_image(self,i):
        self.figure = plt.figure(figsize=(7, 6), dpi=100)
        plt.imshow(self.images[i], cmap='gray')
        canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        canvas.get_tk_widget().grid(row=6, column=3)