import pandas as pd
import numpy as np
import torch
from n_networks.data_augmentation import DataAug
from n_networks.vae import ConvVAE
import matplotlib.pyplot as plt
from coreset1_pu_train import train_coreset1_pu_cnn
from coreset1_pu_train_vae import train_vae
from n_networks.neural_net import MyDataset
from n_networks.vae import MyDatasetV
from torch.utils.data import DataLoader
from n_networks.neural_net import ChatGPT
from active_learning.coreset_functions import active_learning


def rescale(image):
    if np.max(image) == np.min(image):
        return image
    elif np.max(image) == 0:
        return image
    else:
        return (image - np.min(image)) / (np.max(image) - np.min(image))

def coreset(df,current_df):
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
        #
    inter_df = df.iloc[indexes]
    current_df = current_df.append(inter_df)
    df = df.drop(indexes)
    df = df.reset_index(drop=True)


    return current_df, df





def get_highest_pu_samples(df, current_df):
    # reset index of the dataframe
    df = df.reset_index(drop=True)
    data = MyDataset(df)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    # load the model
    model = ChatGPT()
    state_dict = torch.load('models/coreset_pu_CNN_pth')
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
df = pd.read_pickle('C://users/matteo/pycharmprojects/TBProject/labelled_data/smear_2156_17_30.pkl')
for i in range(0, 1):
    df = df.append(pd.read_pickle('C://users/matteo/pycharmprojects/TBProject/labelled_data/smear_2156_17_3' + str(i) + '.pkl'))
# update the index column
df = df.reset_index(drop=True)


# coreset loop
current_df = pd.DataFrame(columns=['image', 'label'])
current_df, df = coreset(df, current_df)
# get first coreset after training the VAE on the whole dataset






# train the model with the 150 samples
# in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu_CNN_1.pt, full_pu_CNN_2.pt, etc.
acc = train_coreset1_pu_cnn(current_df, 0)
accuracy_list.append(acc)

#5000 / 150 = 33 iterations
for i in range(1, 10):
    # get the 150 samples with the highest prediction uncertanty
    current_df, df = get_highest_pu_samples(df, current_df)
    #drop from df last 150 samples of current_df




    # train the model with the 150 samples
    # in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu_CNN_1.pt, full_pu_CNN_2.pt, etc.
    acc = train_coreset_pu_cnn(current_df, i)
    accuracy_list.append(acc)

# plot the accuracy list verus number of data points
plt.plot(accuracy_list, np.arange(0, 5000, 150))
plt.show()
# save the image
plt.savefig('accuracy_list.png')