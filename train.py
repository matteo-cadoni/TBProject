from model.neural_net import Net, ChatGPT, MyDataset, toy_model
from model.autoencoder import Autoencoder_conv, Autoencoder_mlp
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
from torch.utils.tensorboard import SummaryWriter


def arguments_parser():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Tubercolosis Detection')
    parser.add_argument('config', type=str, default='configs/train.yaml',
                        help='configure file for training')
    return parser


def main():
    parser = arguments_parser()
    pars_arg = parser.parse_args()
    
    writer = SummaryWriter()
    
    # set a seed for reproducibility
    torch.manual_seed(42)
    
    # read config as dictionary
    with open(pars_arg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load data from a .pkl file
    loading_config = config['load']
    print("--------------------------------------")
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
    test_eval = train_config['test_eval']
    
    print('Splitting data into train and test and preparing DataLoader')
    
    # split data into train and test and set seed for reproducibility
    np.random.seed(42)
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    dataAug = DataAug(train)
    train = dataAug.augment()
    print('Training data augmented, shape: ', train.shape)
    
    train_dataset = MyDataset(train)
    test_dataset = MyDataset(test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Train data is ready, lenght: ", len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("Test data is ready, lenght: ", len(test_dataset))
    print("--------------------------------------")
    # train model
    #net = Net()
    net = ChatGPT()
    #net = Autoencoder_conv()
    print("Model loaded")
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
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is: ", device)
    
    net = net.to(device)
    print("Model moved to device: ", device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    print("--------------------------------------")
    print("Training started")
    print("--------------------------------------")
    for ep in range(epochs):  # loop over the dataset multiple times
        train_loss = []
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data

            inputs=torch.unsqueeze(inputs,1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
        total_train_loss = (np.sum(train_loss) / len(train_dataset)) * 10000
        writer.add_scalar("Training Loss", total_train_loss, ep)
        
        if ep % test_eval == 0:
            net.eval()
            test_loss = []
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.to(torch.float32)
                    images= images.view(images.shape[0],1,50,50)
                    labels = labels.to(torch.float32)
                    images = images.to(device)
                    labels = labels.to(device)
                
                    outputs = net(images)
                    outputs = outputs.squeeze(1)
                    
                    loss = criterion(outputs, labels)
                    test_loss.append(loss.item())
                    
                    for i, output in enumerate(outputs):
                        if output > 0.7:
                            outputs[i] = 1
                        else:
                            outputs[i] = 0
                    
                    total += labels.size(0)    
                    correct += (outputs == labels).sum().item()
                accuracy = 100 * correct / total
                total_test_loss = (np.sum(test_loss) / len(test_dataset)) * 10000
                
                writer.add_scalar("Test Accuracy", accuracy, ep)
                writer.add_scalar("Test Loss", total_test_loss, ep)

            net.train()
        print("Epoch: {}".format(ep+1))
        print("Train Loss: {:f}".format(total_train_loss))
        if ep % test_eval == 0:
            print("Test Loss: {:f}, Accuracy {:f}".format(total_test_loss, accuracy))
        print("------------------")
            
        
    print('Finished Training')
    #print finished training statistics
    print('Saving model')
    if not os.path.exists('model_ckpt'):
        os.makedirs('model_ckpt')
    torch.save(net.state_dict(), 'model_ckpt/model.pth')
    
    writer.flush()
    writer.close()

    # test model
    #print('Testing model on validation set')
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(torch.float32)
            images= images.view(images.shape[0],1,50,50)
            labels = labels.to(torch.float32)
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            #print("Shape of outputs and labels is: ", outputs.shape, labels.shape)
            #print("outputs: ", outputs)
            outputs = outputs.squeeze(1)
            for i, output in enumerate(outputs):
                if output > 0.5:
                    outputs[i] = 1
                else:
                    outputs[i] = 0
            #print("outputs and shape", outputs, outputs.shape)
            #print("labels and shape", labels, labels.shape)
            total += labels.size(0)
            #print("total: ", total)
            
            correct += (outputs == labels).sum().item()
            #print("correct: ", correct)
            
    print('Final accuracy of the network on the validation set: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    main()