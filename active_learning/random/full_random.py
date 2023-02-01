import pandas as pd
import numpy as np
import torch
from n_networks.data_augmentation import DataAug
import matplotlib.pyplot as plt
from train_random import train_randomly
from n_networks.neural_net import MyDataset
from torch.utils.data import DataLoader
from n_networks.neural_net import ChatGPT



def get_new_dataset(df, current_df):
    # reset index of the dataframe

    # get 150 random samples from the dataset
    m_df = df.sample(n=78)
    # append the 150 samples to the current dataset
    current_df = current_df.append(m_df)
    # remove the 150 samples from the dataset
    df = df.drop(m_df.index)
    df = df.reset_index(drop=True)

    return current_df, df


# Active Learning approach using Prediction uncertanty from start to end

# query size 3 % of the dataset of size 2600 = 78

# initialize accuracy list
accuracy_list = []


# load full dataset
df = pd.read_pickle('D:/dataframe/all2.pkl')
# take 1300 1 and 1300 0, drop the rest
df1 = df[df['label'] == 1].sample(n=1300)
df0 = df[df['label'] == 0].sample(n=1300)
df = df1.append(df0)
print(df.shape)
print(df['label'].value_counts())


df = df.reset_index(drop=True)

# get 150 random samples from the dataset
current_df = df.sample(n=78)

# remove the 150 samples from the dataset
df = df.drop(current_df.index)
df = df.reset_index(drop=True)


# train the model with the 150 samples
# in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu_CNN_1.pt, full_pu_CNN_2.pt, etc.
acc = train_randomly(current_df, 0)
accuracy_list.append(acc)

#2600 / 78 = 33
for i in range(1, 32):
    # get the 150 samples with the highest prediction uncertanty
    current_df, df = get_new_dataset(df, current_df)

    # train the model with the 150 samples
    # in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu_CNN_1.pt, full_pu_CNN_2.pt, etc.
    acc = train_randomly(current_df, i)
    accuracy_list.append(acc)
    print(accuracy_list)
# plot the accuracy list verus number of data points
plt.plot(accuracy_list, np.arange(78, 2600, 78))
plt.show()
# save the image
plt.savefig('accuracy_list.png')