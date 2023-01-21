from model.neural_net import Net, ChatGPT, MyDataset, toy_model
from model.autoencoder import Autoencoder_conv, Autoencoder_mlp, Autoencoder_dataset
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

    # load data from a .pkl file
    loading_config = config['load']
    print('Loading data from pkl file')
    data_paths = loading_config['data_path']
    print('Data paths: ', data_paths)
    data = []
    for i, path in enumerate(data_paths):
        data.append(pd.read_pickle(path))
        print("Added data from path and shape: ", path, data[i].shape)
    data = pd.concat(data)
    print('Data loaded, shape: ', data.shape)

    train_config = config['train']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    type = train_config['type']
    
    # take the first image in data and put it in a new dataframe
    overfit = 0 
    if overfit == 1:
        data = data.iloc[50:51]
        train_dataset = Autoencoder_dataset(data)
        test_dataset = Autoencoder_dataset(data)
    else:
        print('Splitting data into train and test and preparing DataLoader')
        train, test = train_test_split(data, test_size=0.2)
        train_dataset = Autoencoder_dataset(train)
        test_dataset = Autoencoder_dataset(test)
    
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Train data is ready, lenght: ", len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Test data is ready, lenght: ", len(test_dataset))

    # train model
    if type == 'conv':
        net = Autoencoder_conv()
        print("Convolutional autoencoder loaded")

    elif type == 'mlp':
        net = Autoencoder_mlp()
        print("MLP autoencoder loaded")
    
    print(net)
    time.sleep(2)

    print('Initialising weights')
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    net.apply(weights_init)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    loss_epoch = 0
    print("Training starting")
    for ep in range(epochs):  # loop over the dataset multiple times
        train_loss = []
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, _ = data
            #print("Shape of inputs and labels is: ", inputs.shape)
            
            #print("Shape of inputs and labels after unsqueezing is: ", inputs.shape)
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(inputs, outputs)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                current_loss = loss / inputs.shape[0]
                print(f"Epoch: {ep+1}, Batch: {i+1}/{len(train_loader)}, Loss: {current_loss}")
        current_loss = np.sum(train_loss) / len(train_dataset)
        print('Epoch: {}, Loss: {:.4f}'.format(ep+1, current_loss))
        
        if ep % 10 == 0:
            if loss_epoch == np.sum(train_loss) / len(train_dataset):
                pass
        loss_epoch = np.sum(train_loss) / len(train_dataset)
            
    
    print('Finished Training')
    #print finished training statistics
    print('Saving model')
    torch.save(net.state_dict(), 'model_ckpt/model.pth')
    


    # test model
    print('Testing model on validation set')
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        test_loss = 0
        for data in test_loader:
            img, _ = data
            img =torch.unsqueeze(img,1)
            output = net(img)
            loss = criterion(output, img)
            test_loss += loss.item()
        test_loss /= len(test_dataset)
        print('Test Loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    main()