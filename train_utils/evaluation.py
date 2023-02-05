import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, roc_curve, auc


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

def compute_accuracy(net, loader): # not used anymore
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

def get_model_predictions(net, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    predictions = []
    predictions_binarized = []
    with torch.no_grad():
        for data in loader:
            
            image, label = data
            image = image.to(torch.float32)
            image = image.view(image.shape[0],1,50,50)
            label = label.to(torch.float32)
            image = image.to(device)
            label = label.to(device)
            
            output = net(image)
            output = output.squeeze(1)
            
            if output > 0.5:
                binary_output = 1
            else:
                binary_output = 0
                
            predictions.append(output.item())
            predictions_binarized.append(binary_output)

    return predictions, predictions_binarized
        
def get_prediction_matrix(all_predictions):
    # make an empty matrix of rows equal to the number of models and columns equal to the number of predictions
    prediction_matrix = np.empty((0, len(all_predictions[0])))
    for pred in all_predictions:
        # make this list of values a numpy array and save it in a row of a matrix
        pred = np.array(pred)
        prediction_matrix = np.vstack((prediction_matrix, pred))
    return prediction_matrix
       
def model_averaged_predictions(prediction_matrix):
    # for each column, find the average of the values
    averaged_predictions = np.mean(prediction_matrix, axis=0)
    # for each value in the averaged predictions, if it is greater than 0.5, make it 1
    averaged_predictions_binarized = np.where(averaged_predictions > 0.5, 1, 0)
    return averaged_predictions, averaged_predictions_binarized     


def get_model_metrics(labels_test_set, predictions_binarized):
    print("Computing statistics of the model on the test set")
    accuracy = accuracy_score(labels_test_set, predictions_binarized)
    recall = recall_score(labels_test_set, predictions_binarized)
    precision = precision_score(labels_test_set, predictions_binarized)
    f1 = f1_score(labels_test_set, predictions_binarized)
    
    print("Accuracy of model on test set: {:f}".format(accuracy*100))
    print("Recall of model on test set: {:f}".format(recall*100))
    print("Precision of model on test set: {:f}".format(precision*100))
    print("F1 score of model on test set: {:f}".format(f1*100))
    return accuracy, recall, precision, f1

def print_test_metrics(accuracies, recall, precision, f1):
    results = {'Accuracy': accuracies, 'Recall': recall, 'Precision': precision, 'F1': f1}
    results["Mean Accuracy"] = np.mean(accuracies)
    results["Mean Recall"] = np.mean(recall)
    results["Mean Precision"] = np.mean(precision)
    results["Mean F1"] = np.mean(f1)
    results["Std Accuracy"] = np.std(accuracies)
    results["Std Recall"] = np.std(recall)
    results["Std Precision"] = np.std(precision)
    results["Std F1"] = np.std(f1)
    
    print('Average of ACCURACIES of all models on test set: {:f}'.format(np.mean(accuracies)*100))
    print("Standard deviation of ACCURACIES of all models on test set: {:f} \n".format(np.std(accuracies)))
    print('Average of RECALL of all models on test set: {:f}'.format(np.mean(recall)*100))
    print("Standard deviation of RECALL of all models on test set: {:f} \n".format(np.std(recall)))
    print('Average of PRECISION of all models on test set: {:f}'.format(np.mean(precision)*100))
    print("Standard deviation of PRECISION of all models on test set: {:f} \n".format(np.std(precision))) 
    print('Average of F1 SCORE of all models on test set: {:f}'.format(np.mean(f1)*100))
    print("Standard deviation of F1 SCORE of all models on test set: {:f} \n".format(np.std(f1)))
    return results
    
def print_training_statistics(ep, train_loss, test_loss, accuracy):
    print("Epoch: {}".format(ep+1))
    print("Train Loss: {:f}".format(train_loss))
    print("Validation Loss: {:f}, Accuracy {:f}".format(test_loss, accuracy))
    print("------------------")
    
