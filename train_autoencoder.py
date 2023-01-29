from model.neural_net import Net, ChatGPT, MyDataset, toy_model
from model.autoencoder import Autoencoder_conv, Autoencoder_mlp, ImageTensorDataset, Autoencoder_dataset
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
import torchvision.transforms as transforms

def arguments_parser():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Tubercolosis Detection')
    parser.add_argument('config', type=str, default='configs/train.yaml',
                        help='configure file for training')
    return parser

def save_model(model):
    # check if directory model_ckpt exists
    if not os.path.exists('model_ckpt'):
        os.makedirs('model_ckpt')
    torch.save(model.state_dict(), 'model_ckpt/autoencoder.pth')
    print("Model saved")
    
# take data in a dataframe and return a tensor
def dataframe_to_tensor(data):
    
    tensor = torch.stack([torch.Tensor(i[0]) for i in data.values])
    labels = [i[1] for i in data.values]
    assert tensor.shape[0] == len(labels)
    return tensor, labels
    

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
        print("Added data from path: ", path)
        print("Shape of data: ", data[i].shape)
    data = pd.concat(data)
    print('Data loaded, shape: ', data.shape)
    data_tensor, labels_list = dataframe_to_tensor(data)


    train_config = config['train']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    type = train_config['type']
    
    # # take the first image in data and put it in a new dataframe
    # overfit = 0 
    # if overfit == 1:
    #     data = data.iloc[50:51]
    #     train_dataset = Autoencoder_dataset(data)
    #     test_dataset = Autoencoder_dataset(data)
    # else:
    #     print('Splitting data into train and test and preparing DataLoader')
    #     train, test = train_test_split(data, test_size=0.2)
    #     train_dataset = Autoencoder_dataset(train)
    #     test_dataset = Autoencoder_dataset(test)
    
    
    transformation = torch.nn.Compose(
        transforms.hflip()
    )
    # Apply the transforms to the data
    # duplicate the original training data
    
    
    for data in data_tensor:
        transformed_data = scripted_transformation(data)
        breakpoint()
        
    transformed_data = data_transforms(train_dataset)
    print("Transformed data shape: ", transformed_data.shape)

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

    # print('Initialising weights')
    # def weights_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         m.bias.data.fill_(0.01)
    #     elif isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         m.bias.data.fill_(0.01)
    
    # net.apply(weights_init)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is: ", device)
    
    
    net = net.to(torch.device("cuda"))
    print('Model loaded on device')
    
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
            inputs = inputs.to(torch.device("cuda"))
            #print("Shape of inputs and labels after unsqueezing is: ", inputs.shape)
            optimizer.zero_grad()
            
            
            outputs = net(inputs)
            outputs = torch.squeeze(outputs, 1)
            #print("Shape of inputs and outputs is: ", inputs.shape, outputs.shape)
            loss = criterion(inputs, outputs)
            train_loss.append(loss.item())


            loss.backward()
            optimizer.step()
            if i % 1000 == 999:
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
    save_model(net)
    


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