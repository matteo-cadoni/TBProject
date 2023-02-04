from model.neural_net import Net, BacilliNet, MyDataset, toy_model
#from model.autoencoder import Autoencoder_conv, Autoencoder_mlp
from data_augmentation import DataAug
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import argparse
import yaml
import time
import os
import json
#from torch.utils.tensorboard import Summary#writer
from matplotlib import pyplot as plt
from datetime import datetime

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
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # load data from a .pkl file
    loading_config = config['load']
    sampling_percentage = loading_config['sampling_percentage']

    loadr = loader(loading_config)
    data = loadr.load()
    #data = loadr.apply_filters(data)
    time.sleep(2)
    print("--------------------------------------")
    filter_config = config['filter']
    if filter_config['remove_black_img']:
        print("Applying filters to data")
        data, n_removed_imgs = remove_black_img(data)
        print("Removed ", n_removed_imgs, " black images")
    print("Data shape is now: ", data.shape)
    time.sleep(2)
    train_config = config['train']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    n_splits = train_config['n_splits']


    train, test = train_test_split(data, test_size=0.2, random_state=random_state)
    print('Train and test data splitted, train shape: ', train.shape, 'test shape: ', test.shape)
    test_dataset = MyDataset(test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("Test dataloader is ready")
    
    # sample a subset of train set
    print('Sampling train data, with sampling percentage: ', sampling_percentage * 100, '%')
    train_sampled = sample(train, p = sampling_percentage, random_state=random_state)
    print('After sampling, new train dataset shape is ', train_sampled.shape)

    # check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print("Device is: ", device)

    
    # #writer for tensorboard
    #writer = Summary#writer()
    
    criterion = nn.BCELoss()
    
    
    print("Starting cross-validation\n")
    # cross-validation
    fold = 0
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    training_losses =[]
    val_losses = []
    val_acc = []
    test_acc = []
    models = {}
    
    for train_index, val_index in kfold.split(train_sampled):

        time.sleep(1)
        
        # restart model and optimizer at each fold
        net = BacilliNet()
        if fold == 0:
            print("Model loaded")
            print(net)
            time.sleep(2)
            
        net.apply(weights_init)
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        train, val = train_sampled.iloc[train_index], train_sampled.iloc[val_index]
        print('Train and validation data splitted, train shape: ', train.shape, 'validation shape: ', val.shape)       
        
        dataAug = DataAug(train)
        train = dataAug.augment()
        print('Training data augmented, shape: ', train.shape)
    
        train_dataset = MyDataset(train)
        val_dataset = MyDataset(val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print("Train dataloader is ready")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        print("Validation dataloader is ready")
        
        print("--------------------------------------")
        time.sleep(2)
    
        net.train()

        train_loss_fold = []
        val_loss_fold = []
        acc_fold = []
        
        print(f"Starting Training of model {fold+1}")
        print("--------------------------------------")
        for ep in range(epochs):  # loop over the dataset multiple times
            train_loss_epoch = []
            for data in train_loader:
                training_step(data, net, device, optimizer, criterion, train_loss_epoch)
            
            average_train_loss = np.mean(train_loss_epoch)
            
            average_val_loss, accuracy = evaluation_step(net, val_loader, criterion)

            print_training_statistics(ep, average_train_loss, average_val_loss, accuracy) 
                   
            train_loss_fold.append(average_train_loss)
            val_loss_fold.append(average_val_loss)
            acc_fold.append(accuracy)

        print(f'Finished Training of model {fold+1}\n')
        
        print("Computing accuracy of the model on the test set")
        accuracy = compute_accuracy(net, test_loader)
        print("Accuracy of model {} on test set: {:f}".format(fold+1, accuracy))
        
        
        # appending losses and accuracies of current training to a list
        training_losses.append(train_loss_fold)
        val_losses.append(val_loss_fold)
        val_acc.append(acc_fold)
        test_acc.append(accuracy)
        
        fold += 1
        
        time.sleep(0.5)
        models[fold] = net
        
        print("--------------------------------------")
    
    print('Finished cross-validation\n')

    print('Average of accuracies of all models on test set: {:f}'.format(np.mean(test_acc)))
    print("Standard deviation of accuracies of all models on test set: {:f}".format(np.std(test_acc)))
    
    
    
    print('Saving model')
    if not os.path.exists('model_ckpt'):
        os.makedirs('model_ckpt')
    torch.save(net.state_dict(), 'model_ckpt/model.pth')
    
    #writer.flush()
    #writer.close()

    
    # plot train and test loss over all epochs
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    experiment_data = f"{sampling_percentage*100}%_{batch_size}_{current_time}"    
    # create a folder named experiment_data in plot
    if not os.path.exists(f'plots/{experiment_data}'):    
        os.mkdir(f'plots/{experiment_data}')
    save_path = f'plots/{experiment_data}'
    
    fig1 = plt.gcf()
    for i in range(n_splits):
        plt.plot(training_losses[i], label=f"Train loss model {i+1}")
        plt.plot(val_losses[i], label=f"Val loss model {i+1}")
    
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    #plt.legend(['Train Loss', 'Test Loss'])
    plt.title('Losses')
    plt.show()
    plt.draw()
    # save plot to file
    
    fig1.savefig(f'plots/{experiment_data}/losses.png')
    
    # plot accuracies
    fig2 = plt.gcf()
    for i in range(n_splits):
        acc = val_acc[i]
        plt.plot(acc, label=f"Model {i+1}")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on validation set')
    plt.show()

    fig2.savefig(f'plots/{experiment_data}/accuracy.png')
    
    # save the config variable to a json file in the experiment_data folder
    with open(f'plots/{experiment_data}/config.json', 'w') as fp:
        json.dump(config, fp)


if __name__ == '__main__':
    main()