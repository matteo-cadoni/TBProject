import torch.nn as nn
from torch.utils.data import Dataset
import torch

class LImageDiscriminator(nn.Module):
    def __init__(self):
        super(LImageDiscriminator, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        #self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # 3rd convolutional layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        #self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # 4th convolutional layer
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 12 * 12, out_features=512)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=16)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1st convolutional layer
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        # 2nd convolutional layer
        out = self.conv2(out)
        out = self.relu2(out)


        # 3rd convolutional layer
        out = self.conv3(out)
        out = self.relu3(out)

        # 4th convolutional layer
        out = self.conv4(out)
        out = self.relu4(out)

        # Flatten
        out = out.view(out.size(0), -1)


        # Fully connected layers
        out = out.view(-1, 64 * 12 * 12)

        out = self.fc1(out)
        out = self.relu5(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu6(out)
        out = self.sigmoid(out)

        return out

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data.iloc[index]['image']
        # change image values to be in 0,1 range instead of 0, 16000

        label = self.data.iloc[index]['label']
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)