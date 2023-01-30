from n_networks.neural_net import Net, ChatGPT, MyDataset, toy_model
from n_networks.data_augmentation import DataAug
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn

import torch.optim as optim
import torch

from sklearn.model_selection import train_test_split



def train_randomly(dataset, iteration):
    # if iteration is zero start training from scratch else resume training
    if iteration == 0:
        print('Training model on training set')
        # set as batch size 40 % of the dataset size
        batch_size = int(dataset.shape[0] * 0.4)
        epochs = 10
        train, test = train_test_split(dataset, test_size=0.2, random_state=42)
        dataAug = DataAug(train)
        train = dataAug.augment()
        train_dataset = MyDataset(train)
        test_dataset = MyDataset(test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        net = ChatGPT()

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        net.apply(weights_init)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        net.train()
        for ep in range(epochs):  # loop over the dataset multiple times
            print('Epoch: ', ep)
            train_loss = []
            for i, data in enumerate(train_loader):
                # get the inputs
                inputs, labels = data

                inputs = torch.unsqueeze(inputs, 1)
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

        torch.save(net.state_dict(), 'models/full_random_CNN.pth')
        # test model
        # print('Testing model on validation set')
        correct = 0
        total = 0
        net.eval()

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(torch.float32)
                images = images.view(images.shape[0], 1, 50, 50)
                labels = labels.to(torch.float32)
                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                # print("Shape of outputs and labels is: ", outputs.shape, labels.shape)
                # print("outputs: ", outputs)
                outputs = outputs.squeeze(1)
                for i, output in enumerate(outputs):
                    if output > 0.5:
                        outputs[i] = 1
                    else:
                        outputs[i] = 0
                # print("outputs and shape", outputs, outputs.shape)
                # print("labels and shape", labels, labels.shape)
                total += labels.size(0)
                # print("total: ", total)

                correct += (outputs == labels).sum().item()
                # print("correct: ", correct)

        return 100 * correct / total
    else:
        # resume training
        batch_size = int(dataset.shape[0] * 0.4)
        epochs = 10
        train, test = train_test_split(dataset, test_size=0.2, random_state=42)
        dataAug = DataAug(train)
        train = dataAug.augment()
        train_dataset = MyDataset(train)
        test_dataset = MyDataset(test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        net = ChatGPT()
        net.load_state_dict(torch.load('models/full_random_CNN.pth'))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        net.train()
        for ep in range(epochs):  # loop over the dataset multiple times
            print('Epoch: ', ep)
            train_loss = []
            for i, data in enumerate(train_loader):
                # get the inputs
                inputs, labels = data

                inputs = torch.unsqueeze(inputs, 1)
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

        torch.save(net.state_dict(), 'models/full_random_CNN.pth')
        # test model
        # print('Testing model on validation set')
        correct = 0
        total = 0
        net.eval()

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(torch.float32)
                images = images.view(images.shape[0], 1, 50, 50)
                labels = labels.to(torch.float32)
                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                # print("Shape of outputs and labels is: ", outputs.shape, labels.shape)
                # print("outputs: ", outputs)
                outputs = outputs.squeeze(1)
                for i, output in enumerate(outputs):
                    if output > 0.5:
                        outputs[i] = 1
                    else:
                        outputs[i] = 0
                # print("outputs and shape", outputs, outputs.shape)
                # print("labels and shape", labels, labels.shape)
                total += labels.size(0)
                # print("total: ", total)

                correct += (outputs == labels).sum().item()
                # print("correct: ", correct)

        return 100 * correct / total


