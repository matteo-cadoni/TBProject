from model.neural_net import Net, ChatGPT, MyDataset, toy_model
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
    
    print('Splitting data into train and test and preparing DataLoader')
    train, test = train_test_split(data, test_size=0.2)
    train_dataset = MyDataset(train)
    test_dataset = MyDataset(test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Train data is ready, lenght: ", len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Test data is ready, lenght: ", len(test_dataset))

    # train model
    #net = Net()
    net = ChatGPT()
    print("Model loaded")
    print(net)
    #net = toy_model()

    print('Initialising weights')
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    net.apply(weights_init)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    print("Training started")
    for ep in range(epochs):  # loop over the dataset multiple times
        train_loss = []
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            #inputs = inputs.to(torch.float32)

            inputs=torch.unsqueeze(inputs,1)
            #labels = labels.to(torch.float32)
            #labels = labels.to(torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            #print(outputs)
            #breakpoint()
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        print("Epoch: ", ep, " Loss: ", np.sum(train_loss) / len(train_dataset))
    
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
        for data in test_loader:
            images, labels = data
            images = images.to(torch.float32)
            images= images.view(images.shape[0],1,50,50)
            labels = labels.to(torch.float32)
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
            
    print('Accuracy of the network on the validation set: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    main()