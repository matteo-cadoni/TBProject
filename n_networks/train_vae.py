
import torch.optim as optim
import matplotlib
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pandas as pd
from vae import *
from plot_reconstruct import *
from sklearn.model_selection import train_test_split
import os
import time


matplotlib.style.use('ggplot')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the model
model = ConvVAE().to(device)
# set the learning parameters
lr = 0.001
epochs = 200
batch_size = 8
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')
# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
# load pandas dataframe, saved as pkl
my_df = pd.read_pickle('D:/images/smear_2156_17_30.pkl')
for i in range(1, 1345):
    #check if file exists
    if os.path.isfile('D:/images/smear_2156_17_3' + str(i) + '.pkl'):
        my_df = my_df.append(pd.read_pickle('D:/images/smear_2156_17_3' + str(i) + '.pkl'))
#print head
print(my_df.shape)
# show one image
#plt.imshow(my_df.iloc[2200]['image'], cmap='gray')
#plt.show()
#remove all the images but 2200



dataset = MyDataset(my_df)
trainset, testset = train_test_split(dataset, test_size=0.2)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
"""
# training set and train data loader
trainset = torchvision.datasets.MNIST(
    root='../input', train=True, download=True, transform=transform
)

trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)

 #validation set and validation data loader
testset = torchvision.datasets.MNIST(
    root='../input', train=False, download=True, transform=transform
)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)
"""
train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, trainloader, trainset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model, testloader, testset, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

# save the reconstructions as a .gif file
image_to_vid(grid_images)
# save the loss plots to disk
save_loss_plot(train_loss, valid_loss)
print('TRAINING COMPLETE')
