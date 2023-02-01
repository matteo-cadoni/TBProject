import torch
import torch.nn as nn


def training_step(data, net, device, optimizer, criterion, train_loss, total_samples):
    # get the inputs
    inputs, labels = data

    inputs=torch.unsqueeze(inputs,1) #add channel dimension
    inputs = inputs.to(device)
    labels = labels.to(device)
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward + backward + optimize
    outputs = net(inputs)
    outputs = outputs.squeeze(1) #remove channel dimension

    loss = criterion(outputs, labels)
    train_loss.append(loss.item())
    total_samples.append(outputs.shape[0])
    loss.backward()
    optimizer.step()
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        