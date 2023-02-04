import torch
import numpy as np


def evaluation_step(net, loader, criterion):
    net.eval()
    losses = []
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(torch.float32)
            images = images.view(images.shape[0],1,50,50)
            labels = labels.to(torch.float32)
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            for i, output in enumerate(outputs):
                if output > 0.5:
                    outputs[i] = 1
                else:
                    outputs[i] = 0
            
            total += labels.size(0)    
            correct += (outputs == labels).sum().item()
        accuracy = 100 * correct / total
        
        total_loss = np.mean(losses)
        #writer.add_scalar("Test Accuracy", accuracy, ep)
        #writer.add_scalar("Test Loss", total_test_loss, ep)
        net.train()
    return total_loss, accuracy

def compute_accuracy(net, loader):
    net.eval()
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(torch.float32)
            images = images.view(images.shape[0],1,50,50)
            labels = labels.to(torch.float32)
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            outputs = outputs.squeeze(1)
            
            for i, output in enumerate(outputs):
                if output > 0.5:
                    outputs[i] = 1
                else:
                    outputs[i] = 0
            
            total += labels.size(0)    
            correct += (outputs == labels).sum().item()
        accuracy = 100 * correct / total
        
        #writer.add_scalar("Test Accuracy", accuracy, ep)
        #writer.add_scalar("Test Loss", total_test_loss, ep)
        net.train()
    return accuracy


def print_training_statistics(ep, train_loss, test_loss, accuracy):
    print("Epoch: {}".format(ep+1))
    print("Train Loss: {:f}".format(train_loss))
    print("Validation Loss: {:f}, Accuracy {:f}".format(test_loss, accuracy))
    print("------------------")