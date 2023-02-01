import torch
import numpy as np


def evaluation_step(net, test_loader, device, criterion):
    net.eval()
    test_loss = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(torch.float32)
            images= images.view(images.shape[0],1,50,50)
            labels = labels.to(torch.float32)
            images = images.to(device)
            labels = labels.to(device)
        
            outputs = net(images)
            outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels)
            test_loss.append(loss.item())
            
            for i, output in enumerate(outputs):
                if output > 0.5:
                    outputs[i] = 1
                else:
                    outputs[i] = 0
            
            total += labels.size(0)    
            correct += (outputs == labels).sum().item()
        accuracy = 100 * correct / total
        #total_test_loss = (np.sum(test_loss) / len(test_dataset)) * 10000
        total_test_loss = np.mean(test_loss) * 1000
        #writer.add_scalar("Test Accuracy", accuracy, ep)
        #writer.add_scalar("Test Loss", total_test_loss, ep)
        net.train()
    return total_test_loss, accuracy


def print_statistics(ep, train_loss, test_loss, accuracy):
    print("Epoch: {}".format(ep+1))
    print("Train Loss: {:f}".format(train_loss))
    print("Test Loss: {:f}, Accuracy {:f}".format(test_loss, accuracy))
    print("------------------")