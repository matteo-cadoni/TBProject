import time

import pandas as pd
import numpy as np
import torch
from n_networks.data_augmentation import DataAug
import matplotlib.pyplot as plt
from full_pu_cnn_train import train_pu_cnn
from n_networks.neural_net import MyDataset
from torch.utils.data import DataLoader
from n_networks.neural_net import ChatGPT
import os



def get_highest_pu_samples(df, current_df):
    # reset index of the dataframe
    df = df.reset_index(drop=True)
    data = MyDataset(df)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    # load the model
    model = ChatGPT()
    state_dict = torch.load('models/full_pu_CNN.pth')
    model.load_state_dict(state_dict)
    # get the prediction uncertanty for the whole dataset
    predictions = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(torch.float32)
            images = images.view(images.shape[0], 1, 50, 50)
            labels = labels.to(torch.float32)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # print("Shape of outputs and labels is: ", outputs.shape, labels.shape)
            # print("outputs: ", outputs)
            outputs = outputs.squeeze(1)
            for i, output in enumerate(outputs):
                predictions.append(np.array(output))

    # turn torch tensor into numpy array

    for j in range(0, len(predictions)):
        distance = np.abs(0.5, predictions[j])
        predictions[j] = 1 - distance
    # oreder the predictions
    sort = np.argsort(predictions)
    # get the 150 samples with the highest prediction uncertanty
    current_df = current_df.append(df.iloc[sort[-150:]])
    # delete the last 150 samples from the dataset
    df = df.drop(df.iloc[sort[-150:]].index)
    df = df.reset_index(drop=True)

    return current_df, df


# Active Learning approach using Prediction uncertanty from start to end

# query size 3 % of the dataset size 5000 ---> 150

# initialize accuracy list
accuracy_list = []









# load full dataset
# load full dataset
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
current_df = df.sample(n=78, random_state=1)

# remove the 150 samples from the dataset
df = df.drop(current_df.index)
df = df.reset_index(drop=True)


# train the model with the 150 samples
# in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu_CNN_1.pt, full_pu_CNN_2.pt, etc.
acc = train_pu_cnn(current_df, 0)
accuracy_list.append(acc)

#5000 / 150 = 33 iterations
for i in range(1, 32):

    # get the 150 samples with the highest prediction uncertanty
    current_df, df = get_highest_pu_samples(df, current_df)


    # train the model with the 150 samples
    # in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu_CNN_1.pt, full_pu_CNN_2.pt, etc.
    acc = train_pu_cnn(current_df, i)
    accuracy_list.append(acc)
    print("Accuracy list: ", accuracy_list)

# plot the accuracy list verus number of data points
plt.plot(accuracy_list, np.arange(78, 2600, 78))
plt.show()
# save the image
plt.savefig('accuracy_list.png')











