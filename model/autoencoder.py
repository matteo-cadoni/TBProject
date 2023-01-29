import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

# Define the autoencoder model
class Autoencoder_conv(nn.Module):
    def __init__(self):
        super(Autoencoder_conv, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 12, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        # change shape of input image from (batch_size, 50, 50) to (batch_size, 50*50)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder_mlp(nn.Module):
    def __init__(self):
        super(Autoencoder_mlp, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(50*50, 50*50))
        self.decoder = nn.Sequential(
            nn.Linear(50*50, 50*50))
            
            
    # def __init__(self):
    #     super(Autoencoder_mlp, self).__init__()
    #     self.encoder = nn.Sequential(
    #         nn.Linear(50*50, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 128),
    #     )
    #     self.decoder = nn.Sequential(
    #         nn.Linear(128, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 50*50),
    #     )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 50, 50)
        return x


class ImageTensorDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.shape[0]
        

class Autoencoder_dataset(Dataset):
    def __init__(self, data):
        
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img = self.data.iloc[index]['image']
        label = self.data.iloc[index]['label']
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
    def append(self, img):
        self.data = torch.stack(self.data, img)