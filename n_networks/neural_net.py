import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 1)
        self.conv3 = nn.Conv2d(16, 32, 1)
        self.conv4 = nn.Conv2d(32, 64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(64 * 3 * 3, 300)
        self.fc2 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))

        x = x.view(-1, 64 * 3 * 3)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x


class ChatGPT(nn.Module):
    def __init__(self):
        super(ChatGPT, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 12 * 12, out_features=512)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1st convolutional layer
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        # 2nd convolutional layer
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # Fully connected layers
        out = out.view(-1, 64 * 12 * 12)

        out = self.fc1(out)
        out = self.relu3(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out

    def get_feature_vector(self, x):
        """
        Returns the feature vector of the input image, in second last layer
        """
        # 1st convolutional layer
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        # 2nd convolutional layer
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # Fully connected layers
        out = out.view(-1, 64 * 12 * 12)

        out = self.fc1(out)
        out = self.relu3(out)
        # out = self.dropout(out)

        return out


class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()

        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 50 * 50, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.convolution1(x)
        out = self.relu(out)

        out = out.view(-1, 32 * 50 * 50)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# dataloaders for train and test
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data.iloc[index]['image']
        # change image values to be in 0,1 range instead of 0, 16000
        img = img / 16000

        label = self.data.iloc[index]['label']
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
