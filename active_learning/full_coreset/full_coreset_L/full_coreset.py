import pandas as pd
import numpy as np
import torch
from n_networks.data_augmentation import DataAug
from n_networks.vae import ConvVAE
import matplotlib.pyplot as plt
from full_coreset_train import train_coreset_cnn
from full_coreset_train_vae import train_vae
from n_networks.neural_net import MyDataset
from n_networks.vae import MyDatasetV
from torch.utils.data import DataLoader
from n_networks.neural_net import ChatGPT
from active_learning.coreset_functions import active_learning

def get_first_coreset(df):
    #train VAE on the whole dataset

    train_vae(df)

    # load vae
    vae = ConvVAE()
    vae.load_state_dict(torch.load('models/vae.pth'))

    data = MyDatasetV(df)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    # get the feature vector for the whole dataset
    vectors = np.zeros((df.shape[0], 16))     #TODO change to 32 or 16
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image = data

            vectors[i] = vae.generate_feature_vector(image)
    # get the coreset
    acle = active_learning(vectors, 10, 10)

    dist, centers, indexes = acle.robust_k_center()

    return indexes





def get_coreset(df, current_df, iteration, c_indexes):
    # reset index of the dataframe


    data = MyDataset(df)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    # load the model
    model = ChatGPT()
    state_dict = torch.load('models/full_pu_CNN_' + str(iteration) + '.pth')
    model.load_state_dict(state_dict)

    # get the feature vector for the whole dataset
    vectors = np.zeros((df.shape[0], 512))     #TODO change to 32 or 16
    for i, data in enumerate(dataloader):
        images, labels = data
        images = images.to(torch.float32)
        images = images.view(images.shape[0], 1, 50, 50)


        vectors[i] = model.get_feature_vector(images)



    acle = active_learning(vectors, 10 * iteration, c_indexes)
    dist, centers, indexes = acle.robust_k_center()
    current_df = current_df.append(df.iloc[indexes])

    return current_df, df, indexes


# Active Learning approach using Prediction uncertanty from start to end

# query size 3 % of the dataset size 5000 ---> 150

# initialize accuracy list
accuracy_list = []


# load full dataset
df = pd.read_pickle('C://users/matteo/pycharmprojects/TBProject/labelled_data/smear_2156_17_30.pkl')
for i in range(0, 1):
    df = df.append(pd.read_pickle('C://users/matteo/pycharmprojects/TBProject/labelled_data/smear_2156_17_3' + str(i) + '.pkl'))
# update the index column
df = df.reset_index(drop=True)

# get first coreset after training the VAE on the whole dataset
c_indexes = get_first_coreset(df) #TODO change to vae

current_df = df.iloc[c_indexes]
# drop the samples from the original dataset
df = df.drop(c_indexes)
#reset
df = df.reset_index(drop=True)





# train the model with the 150 samples
# in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu_CNN_1.pt, full_pu_CNN_2.pt, etc.
acc = train_coreset_cnn(current_df, 0)
accuracy_list.append(acc)

#5000 / 150 = 33 iterations
for i in range(1, 10):
    # get the 150 samples with the highest prediction uncertanty
    current_df, df, c_indexes = get_coreset(df, current_df, i-1, c_indexes)
    #drop from df last 150 samples of current_df
    df = df.drop(current_df[-150:].index)
    df = df.reset_index(drop=True)



    # train the model with the 150 samples
    # in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu_CNN_1.pt, full_pu_CNN_2.pt, etc.
    acc = train_coreset_cnn(current_df, i)
    accuracy_list.append(acc)

# plot the accuracy list verus number of data points
plt.plot(accuracy_list, np.arange(0, 5000, 150))
plt.show()
# save the image
plt.savefig('accuracy_list.png')