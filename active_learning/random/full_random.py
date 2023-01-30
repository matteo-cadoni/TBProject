import pandas as pd
import numpy as np
import torch
from n_networks.data_augmentation import DataAug
import matplotlib.pyplot as plt
from train_random import train_randomly
from n_networks.neural_net import MyDataset
from torch.utils.data import DataLoader
from n_networks.neural_net import ChatGPT



def get_new_dataset(df, current_df, iteration):
    # reset index of the dataframe
    df = df.reset_index(drop=True)
    # get 150 random samples from the dataset
    m_df = df.sample(n=150)
    # append the 150 samples to the current dataset
    current_df = current_df.append(m_df)
    # remove the 150 samples from the dataset
    df = df.drop(m_df.index)

    return current_df, df


# Active Learning approach using Prediction uncertanty from start to end

# query size 3 % of the dataset size 5000 ---> 150

# initialize accuracy list
accuracy_list = []


# load full dataset
df = pd.read_pickle('C://users/matteo/pycharmprojects/TBProject/labelled_data/smear_2156_17_30.pkl')
for i in range(0, 8):
    df = df.append(pd.read_pickle('C://users/matteo/pycharmprojects/TBProject/labelled_data/smear_2156_17_3' + str(i) + '.pkl'))
# update the index column
df = df.reset_index(drop=True)

# get 150 random samples from the dataset
current_df = df.sample(n=150)

# remove the 150 samples from the dataset
df = df.drop(current_df.index)


# train the model with the 150 samples
# in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu_CNN_1.pt, full_pu_CNN_2.pt, etc.
acc = train_randomly(current_df, 0)
accuracy_list.append(acc)

#5000 / 150 = 33 iterations
for i in range(1, 10):
    # get the 150 samples with the highest prediction uncertanty
    current_df, df = get_new_dataset(df, current_df, i-1)

    # train the model with the 150 samples
    # in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu_CNN_1.pt, full_pu_CNN_2.pt, etc.
    acc = train_randomly(current_df, i)
    accuracy_list.append(acc)

# plot the accuracy list verus number of data points
plt.plot(accuracy_list, np.arange(0, 5000, 150))
plt.show()
# save the image
plt.savefig('accuracy_list.png')