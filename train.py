from model.neural_net import Net, ChatGPT, MyDataset, toy_model
#from model.autoencoder import Autoencoder_conv, Autoencoder_mlp
from data_augmentation import DataAug
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import yaml
import time
import os
#from torch.utils.tensorboard import Summary#writer
from matplotlib import pyplot as plt

from train_utils.load import *
from train_utils.filters import *
from train_utils.evaluation import *
from train_utils.train import *


# randomly sampled subset of input dataframe with sampling probability p
def sample(df, p, random_state=42):
    df_sampled = df.sample(frac=p, random_state=random_state)
    return df_sampled

def arguments_parser():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Tubercolosis Detection')
    parser.add_argument('config', type=str, default='configs/train.yaml',
                        help='configure file for training')
    return parser


def main():
    parser = arguments_parser()
    pars_arg = parser.parse_args()
    
    # read config as dictionary
    with open(pars_arg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # seed for reproducibility
    random_state = config['random_state']

    # set a seed for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # load data from a .pkl file
    loading_config = config['load']
    sampling_percentage = loading_config['sampling_percentage']

    loadr = loader(loading_config)
    data = loadr.load()
    data = loadr.apply_filters(data)
    time.sleep(2)
    print("--------------------------------------")
    print("Applying filters to data")
    filter_config = config['filter']
    # if filter_config['remove_black_img']:
    #     print("Removing black images")
    #     data, n_removed_imgs = remove_black_img(data)
    print("Removed ", n_removed_imgs, " black images")
    print("Data shape is now: ", data.shape)
    time.sleep(2)
    train_config = config['train']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']

    
    print('Splitting data into train and test and preparing DataLoader')

    
    train, test = train_test_split(data, test_size=0.2, random_state=random_state)
    print('Train and test data splitted, train shape: ', train.shape, 'test shape: ', test.shape)
    # sample a subset of train set
    print('Sampling train data, with sampling percentage: ', sampling_percentage * 100, '%')
    train_sampled = sample(train, p = sampling_percentage, random_state=random_state)
    print('After sampling, new train dataset shape is ', train_sampled.shape)
    dataAug = DataAug(train_sampled)
    train = dataAug.augment()
    print('Training data augmented, shape: ', train.shape)
    
    train_dataset = MyDataset(train)
    test_dataset = MyDataset(test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Train data is ready, lenght: ", len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("Test data is ready, lenght: ", len(test_dataset))
    print("--------------------------------------")
    time.sleep(2)
    net = ChatGPT()
    print("Model loaded")
    print(net)
    time.sleep(2)

    print('Initialising weights')
    net.apply(weights_init)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        device = mps_device
    print("Device is: ", device)
    
    net = net.to(device)
    print("Model moved to device: ", device)
    
    # #writer for tensorboard
    #writer = Summary#writer()
    
    
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    
    net.train()
    tr_loss =[]
    tst_loss = []
    acc = []
    print("--------------------------------------")
    print("Training started")
    print("--------------------------------------")
    for ep in range(epochs):  # loop over the dataset multiple times
        train_loss = []
        total_samples = []
        for i, data in enumerate(train_loader):
            #perform training step
            training_step(net, data, device, optimizer, criterion, train_loss, total_samples)
        
        #compute average train loss for this epoch
        average_train_loss = np.mean(train_loss) * 1000
        
        total_test_loss, accuracy = evaluation_step(net, test_loader, device, criterion)

        print_statistics(ep, average_train_loss, total_test_loss, accuracy)        
        tr_loss.append(average_train_loss)
        tst_loss.append(total_test_loss)
        acc.append
        print("------------------")
            
        
    print('Finished Training')
    
    print('Saving model')
    if not os.path.exists('model_ckpt'):
        os.makedirs('model_ckpt')
    torch.save(net.state_dict(), 'model_ckpt/model.pth')
    
    #writer.flush()
    #writer.close()

    evaluation_step(net, test_loader, device, criterion)
    
    print("Best accuracy is: ", max(acc))
    # plot train and test loss over all epochs
    
    plt.plot(tr_loss)
    plt.plot(tst_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.title('Loss over all epochs')
    plt.show()
    
    # plot test loss and accuracy over all epochs
    plt.plot(acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over all epochs')
    plt.show()
        
        
    


if __name__ == '__main__':
    main()