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

    
    
    #load data
    print('Loading data')
    data = np.load('cropped_images_real_674.npy')
    print("Shape of data: ", data.shape)
    print("Number of images: ", data.shape[0])
    #labels_673=np.array([1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,
                #        1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                #        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,0,
                #       0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,
                #        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0])

    labels_674=np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,0,1,0,0,1,
                         0,0,0,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,0,0,1,1,1,0,1,0,0,1,
                         1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,
                         1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                         0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,1,0,
                         1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,
                         1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,
                         0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,
                         1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,1,1,0,
                         1,1,0,1,1,1,1,1,0,1,1])
    
    
    
    #create pandas dataframe
    print('Creating dataframe')
    df = pd.DataFrame()

    
    for i in range( data.shape[0]):

        d={'image':[data[i,:,:]],'label':labels_674[i]}

        df2=pd.DataFrame(d)

        df=pd.concat([df,df2],ignore_index=True)
            
    print('Dataframe shape: ', df.shape)
    print('Number of images in the Dataframe: ', df.shape[0])

    time.sleep(1)
    #split data into train and test with pandas

    train, test = train_test_split(df, test_size=0.2)
    batch_size = config['batch_size']
    epochs = config['epochs']
        
    train_dataset = MyDataset(train)
    test_dataset = MyDataset(test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # train model
    #net = Net()
    net = ChatGPT()
    #net = toy_model()
    #initialise weights
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
        
    
                
        
                



    # test model
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
            
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    main()