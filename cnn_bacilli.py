#get matrix from .npy file
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#cnn for images with 1 channel 50x50 pixels
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#dataloaders for train and test
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data.iloc[index]['image']
        label = self.data.iloc[index]['label']
        return img, label

if __name__ == '__main__':
        #load data
        data = np.load('cropped_images_real.npy')
        labels=np.array([1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,
                         1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                         1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,0,
                         0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,
                         1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0])

        #create pandas dataframe
        df = pd.DataFrame()


        for i in range(data.shape[0]):



            d={'image':[data[i,:,:]],'label':labels[i]}

            df2=pd.DataFrame(d)

            df=pd.concat([df,df2],ignore_index=True)

#split data into train and test with pandas

        train, test = train_test_split(df, test_size=0.2)

        train_dataset = MyDataset(train)
        test_dataset = MyDataset(test)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



        # train model
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        train_loss = []
        for epoch in range(5):  # loop over the dataset multiple times
            train_loss = 0.0
            net.train()
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data
                inputs = torch.tensor(inputs, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.long)
                #inputs = inputs.unsqueeze(1)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # print statistics
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, train_loss / 10))
                    print(train_loss)



        # test model
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = torch.tensor(images, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.long)
                #images = images.unsqueeze(1)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))
















