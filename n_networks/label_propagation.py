import pandas as pd
import torch
from geomloss import SamplesLoss
import numpy as np

def rescale(image):
    if np.max(image) == np.min(image):
        return image
    else:
        return (image - np.min(image)) / (np.max(image) - np.min(image))

class LabelPropagation:
    def __init__(self, labelled_dataset, unlabelled_dataset, k):
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.k = k

    def ot_propagation(self):
        loss = SamplesLoss("sinkhorn", p=2, blur=.05, scaling=.95)
        # get those images that have prediction 1
        bacilli = self.labelled_dataset[self.labelled_dataset['label'] == 1]
        # get those images that have prediction 0
        non_bacilli = self.labelled_dataset[self.labelled_dataset['label'] == 0]


        # distance from bacilli
        bacilli_distance = np.zeros((bacilli.shape[0], self.unlabelled_dataset.shape[0]))
        for i in range(bacilli.shape[0]):
            print(i)
            for j in range(self.unlabelled_dataset.shape[0]):
                if i != j:
                    bacilli_distance[i, j] = loss(torch.from_numpy(rescale(bacilli.iloc[i]['image'])),
                                              torch.from_numpy(rescale(self.unlabelled_dataset.iloc[j]['image'])))
        breakpoint()
        non_bacilli_distance = np.zeros((non_bacilli.shape[0], self.unlabelled_dataset.shape[0]))
        for i in range(non_bacilli.shape[0]):
            for j in range(self.unlabelled_dataset.shape[0]):
                if i != j:
                    non_bacilli_distance[i, j] = loss(torch.from_numpy(rescale(non_bacilli.iloc[i]['image'])),
                                              torch.from_numpy(rescale(self.unlabelled_dataset.iloc[j]['image'])))
        # get k biggest numbers of a matrix

        bacilli_index = np.argpartition(bacilli_distance.ravel(), -self.k)[-self.k:]
        non_bacilli_index = np.argpartition(non_bacilli_distance.ravel(), -self.k)[-self.k:]
        bacilli_index = np.unravel_index(bacilli_index, bacilli_distance.shape)[1]
        non_bacilli_index = np.unravel_index(non_bacilli_index, non_bacilli_distance.shape)[1]

        bacilli_dataset = pd.DataFrame()
        for index in bacilli_index:
            d = {'image': [self.unlabelled_dataset.iloc[index]['image']], 'label': [1]}
            df = pd.DataFrame(d)
            bacilli_dataset = pd.concat([bacilli_dataset, df], ignore_index=True)
        non_bacilli_dataset = pd.DataFrame()

        for index in non_bacilli_index:
            d = {'image': [self.unlabelled_dataset.iloc[index]['image']], 'label' : [0]}
            df = pd.DataFrame(d)
            non_bacilli_dataset = pd.concat([non_bacilli_dataset, df], ignore_index=True)
        new_dataset = pd.concat([bacilli_dataset, non_bacilli_dataset], ignore_index=True)

        return new_dataset

    def l2_propagation(self):

        # get those images that have prediction 1
        bacilli = self.labelled_dataset[self.labelled_dataset['label'] == 1]
        # get those images that have prediction 0
        non_bacilli = self.labelled_dataset[self.labelled_dataset['label'] == 0]

        # distance from bacilli
        bacilli_distance = np.zeros((bacilli.shape[0], self.unlabelled_dataset.shape[0]))
        for i in range(bacilli.shape[0]):
            print(i)
            for j in range(self.unlabelled_dataset.shape[0]):
                if i != j:
                    bacilli_distance[i, j] = np.linalg.norm(bacilli.iloc[i]['image'] - self.unlabelled_dataset.iloc[j]['image'])

        non_bacilli_distance = np.zeros((non_bacilli.shape[0], self.unlabelled_dataset.shape[0]))
        for i in range(non_bacilli.shape[0]):
            for j in range(self.unlabelled_dataset.shape[0]):
                if i != j:
                    non_bacilli_distance[i, j] = np.linalg.norm(non_bacilli.iloc[i]['image'] - self.unlabelled_dataset.iloc[j]['image'])

        bacilli_index = np.argpartition(bacilli_distance.ravel(), -self.k)[-self.k:]
        non_bacilli_index = np.argpartition(non_bacilli_distance.ravel(), -self.k)[-self.k:]
        bacilli_index = np.unravel_index(bacilli_index, bacilli_distance.shape)[1]
        non_bacilli_index = np.unravel_index(non_bacilli_index, non_bacilli_distance.shape)[1]

        bacilli_dataset = pd.DataFrame()
        for index in bacilli_index:
            d = {'image': [self.unlabelled_dataset.iloc[index]['image']], 'label': [1]}
            df = pd.DataFrame(d)
            bacilli_dataset = pd.concat([bacilli_dataset, df], ignore_index=True)
        non_bacilli_dataset = pd.DataFrame()

        for index in non_bacilli_index:
            d = {'image': [self.unlabelled_dataset.iloc[index]['image']], 'label': [0]}
            df = pd.DataFrame(d)
            non_bacilli_dataset = pd.concat([non_bacilli_dataset, df], ignore_index=True)
        new_dataset = pd.concat([bacilli_dataset, non_bacilli_dataset], ignore_index=True)

        return new_dataset




