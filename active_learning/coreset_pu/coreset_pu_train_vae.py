import torch.optim as optim
from torch.utils.data import DataLoader
from active_learning.base_models.vae import *
from sklearn.model_selection import train_test_split


def train_vae(df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvVAE().to(device)
    lr = 0.001
    epochs = 5
    batch_size = 100
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='sum')
    dataset = MyDatasetV(df)
    trainset, testset = train_test_split(dataset, test_size=0.2)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    train_loss = []

    for epoch in range(epochs):
        train_epoch_loss = train(model, trainloader, testset, device, optimizer, criterion)
        train_loss.append(train_epoch_loss)

    torch.save(model.state_dict(), 'models/vae.pth')

