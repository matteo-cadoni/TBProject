from model.neural_net import Net, BacilliNet, MyDataset, toy_model
#from model.autoencoder import Autoencoder_conv, Autoencoder_mlp
from data_augmentation import DataAug
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import argparse
import yaml
import time
import os
import json
#from torch.utils.tensorboard import Summary#writer
from matplotlib import pyplot as plt
from datetime import datetime

from train_utils.load import *
from train_utils.filters import *
from train_utils.evaluation import *
from train_utils.train import *
from train_utils.plot_new import *

from matplotlib.ticker import PercentFormatter
import seaborn as sns


def cm_analysis(y_true, y_pred, filename, labels, classes, ymap=None, figsize=(17,17)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      classes:   aliases for the labels. String array to be shown in the cm plot.
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    sns.set(font_scale=2.8)

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            #if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
            #elif c == 0:
            #    annot[i, j] = ''
            #else:
            #    annot[i, j] = '%.2f%%\n%d' % (p, c)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    fig, ax = plt.subplots(figsize=figsize)
    plt.yticks(va='center')

    sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=True, cbar_kws={'format':PercentFormatter()}, yticklabels=classes, cmap="Reds")
    plt.savefig(filename,  bbox_inches='tight')
    # close figure 
    plt.close(fig)


# randomly sampled subset of input dataframe with sampling probability p
def sample(df, p, random_state=42):
    df_sampled = df.sample(frac=p, random_state=random_state)
    return df_sampled

def arguments_parser():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Tubercolosis Detection')
    parser.add_argument('config', type=str, default='configs/train.yaml',
                        help='configure file for training')
    return parser


def main():
    parser = arguments_parser()
    pars_arg = parser.parse_args()
    
    # read config as dictionary
    with open(pars_arg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # seed for reproducibility
    random_state = config['random_state']
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # load data from a .pkl file
    loading_config = config['load']
    sampling_percentage = loading_config['sampling_percentage']

    loadr = loader(loading_config)
    data = loadr.load()
    
    
    print("--------------------------------------")
    data = loadr.apply_filters(data)
    
    time.sleep(0.5)
    print("--------------------------------------")
    

    print("Data shape is now: ", data.shape)
    time.sleep(0.5)
    train_config = config['train']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    n_splits = train_config['n_splits']
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # get current date
    today = datetime.today()
    day = today.strftime("%b-%d-%Y")
    
    save_folder_name = f"{current_time}_{day}_sampling-{sampling_percentage*100}%_batchsize-{batch_size}"    
    
    


    train, test = train_test_split(data, test_size=0.2, random_state=random_state)
    print('Train and test data splitted, train shape: ', train.shape, 'test shape: ', test.shape)
    test_dataset = MyDataset(test)
    labels_test_set = test_dataset.data["label"].values
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("Test dataloader is ready")
    
    # sample a subset of train set
    print('Sampling train data, with sampling percentage: ', sampling_percentage * 100, '%')
    train_sampled = sample(train, p = sampling_percentage, random_state=random_state)
    print('After sampling, new train dataset shape is ', train_sampled.shape)

    # check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print("Device is: ", device)

    
    # #writer for tensorboard
    #writer = Summary#writer()
    
    criterion = nn.BCELoss()
    
    print("--------------------------------------")
    print("Starting cross-validation\n")
    print("Splitting train data into ", n_splits, " folds")
    # cross-validation
    fold = 0
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state, )
    training_losses =[]
    val_losses = []
    val_acc = []
    test_acc = []
    test_recall = []
    test_precision = []
    test_f1 = []
    all_predictions = []
    all_predictions_binarized = []
    models = {}
    
    for train_index, val_index in kfold.split(train_sampled):

        time.sleep(1)
        
        # restart model and optimizer at each fold
        net = BacilliNet()
        if fold == 0:
            print("Model loaded")
            print(net)
            time.sleep(2)
            
        net.apply(weights_init)
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        train, val = train_sampled.iloc[train_index], train_sampled.iloc[val_index]
        print('Train and validation data splitted, train shape: ', train.shape, 'validation shape: ', val.shape)       
        
        dataAug = DataAug(train)
        train = dataAug.augment()
        print('Training data augmented, shape: ', train.shape)
    
        train_dataset = MyDataset(train)
        val_dataset = MyDataset(val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print("Train dataloader is ready")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        print("Validation dataloader is ready")
        
        print("--------------------------------------")
        time.sleep(2)
    
        net.train()

        train_loss_fold = []
        val_loss_fold = []
        acc_fold = []
        
        print(f"Starting Training of model {fold+1}")
        print("--------------------------------------")
        
        for ep in range(epochs):  # loop over the dataset multiple times
            train_loss_epoch = []
            for data in train_loader:
                training_step(data, net, device, optimizer, criterion, train_loss_epoch)
            
            average_train_loss = np.mean(train_loss_epoch)
            
            average_val_loss, val_accuracy = evaluation_step(net, val_loader, criterion)

            print_training_statistics(ep, average_train_loss, average_val_loss, val_accuracy) 
                   
            train_loss_fold.append(average_train_loss)
            val_loss_fold.append(average_val_loss)
            acc_fold.append(val_accuracy)


        print(f'Finished Training of model {fold+1}\n')
        
        # get model predictions on test set
        predictions, predictions_binarized = get_model_predictions(net, test_loader)
        
        accuracy, recall, precision, f1 = get_model_metrics(predictions_binarized, labels_test_set)
        
        
        # appending information of current training to a list
        training_losses.append(train_loss_fold)
        val_losses.append(val_loss_fold)
        val_acc.append(acc_fold)
        all_predictions.append(predictions)
        all_predictions_binarized.append(predictions_binarized)
        test_acc.append(accuracy)
        test_recall.append(recall)
        test_precision.append(precision)
        test_f1.append(f1)
        
        fold += 1
        
        models[fold] = net
        
        print("--------------------------------------")
        time.sleep(0.5)
    
    print('Finished cross-validation\n')

    metrics_on_test = print_test_metrics(test_acc, test_recall, test_precision, test_f1)
    
    prediction_matrix_binarized = get_prediction_matrix(all_predictions_binarized)
    prediction_matrix = get_prediction_matrix(all_predictions)
    averaged_predictions, averaged_predictions_binarized = model_averaged_predictions(prediction_matrix)
    assert len(averaged_predictions) == len(labels_test_set), "Average predictions and labels have different length"
    
    
    #print('Saving model')
    #if not os.path.exists('model_ckpt'):
     #   os.makedirs('model_ckpt')
    #torch.save(net.state_dict(), 'model_ckpt/model.pth')
    
    #writer.flush()
    #writer.close()

    
    # PLOT AND SAVE ----------------------------------------------------------------------------------------------
    
    results_config = config['results']
    
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # get current date
    today = datetime.today()
    day = today.strftime("%b-%d-%Y")
    
    save_folder_name = f"{current_time}_{day}_sampling-{sampling_percentage*100}%_batchsize-{batch_size}"    
    # create a folder named save_folder_name in plot
    if not os.path.exists(f'plots/{save_folder_name}'): 
        os.makedirs(f'plots/{save_folder_name}/accuracies')
        os.makedirs(f'plots/{save_folder_name}/losses')
        os.makedirs(f'plots/{save_folder_name}/confusion_matrices')
        os.makedirs(f'plots/{save_folder_name}/models')
        os.makedirs(f'plots/{save_folder_name}/results')
    save_path = f'plots/{save_folder_name}'
    print("Created folder for plots named ", save_path)
    
    # reset rcParams
    #plt.rcParams.update(plt.rcParamsDefault)
    
    
    # save models
    if results_config['save_models']:
        for i in range(n_splits):
            torch.save(models[i+1].state_dict(), f'plots/{save_folder_name}/models/model_{i+1}.pth')
            print("Saved model", i+1, "in plots folder")
        
    # plot losses
    for i in range(n_splits):
        fig = plt.gcf()
        plt.plot(training_losses[i], label=f"Train loss model {i+1}")
        plt.plot(val_losses[i], label=f"Val loss model {i+1}")    
        plt.xlabel('Epochs')
        plt.ylabel('BCE Loss')
        plt.legend(['Train Loss', 'Test Loss'])
        plt.title('Losses')
        plt.show()
        plt.draw()
        # save plot to file
        fig.savefig(f'plots/{save_folder_name}/losses/loss_model_{i+1}.png')
        # close figure
        plt.close(fig)
    print('Loss plots saved in plots folder')
    
    # plot accuracies
    for i in range(n_splits):
        fig = plt.gcf()
        acc = val_acc[i]
        plt.plot(acc, label=f"Model {i+1}")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy on validation set')
        plt.show()
        fig.savefig(f'plots/{save_folder_name}/accuracies/accuracy_model_{i+1}.png')
        plt.close(fig)
    print('Accuracy plots saved in plots folder')
    
    # compute and plot confusion matrix of averaged predictions
    # count the number of bacilli in the test set
    bacilli_in_test = 0
    for i in range(len(labels_test_set)):
        if labels_test_set[i] == 1:
            bacilli_in_test += 1
    print(f"Number of bacilli in test set: {bacilli_in_test}")
    print(f"Number of non-bacilli in test set: {len(labels_test_set) - bacilli_in_test}")
    
    # compute and plot confusion matrix of each model
    for i in range(n_splits):
        cm = confusion_matrix(labels_test_set, all_predictions_binarized[i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        disp.plot()
        plt.show()
        plt.draw()
        fig = plt.gcf()
        fig.savefig(f'plots/{save_folder_name}/confusion_matrices/confusion_matrix_model_{i+1}.png')
        # close figure 
        plt.close(fig)
    print('Confusion matrices for each model saved in plots folder')    
    
    # unified plot of accuracies and losses
    plot_validation_accuracies(val_acc, save_folder_name)    
    print('Plot of validation accuracies of different models saved in plots folder')
    plot_training_and_validation_losses(training_losses, val_losses, save_folder_name)
    print("Plot of training and validation losses of different models saved in plots folder")
    
    #save results
    # save results of each model in a csv file, including losses, accuracy and metrics
    for i in range(n_splits):
        results = pd.DataFrame({'loss': val_losses[i], 'accuracy': val_acc[i]})
        results.to_csv(f'plots/{save_folder_name}/results/results_model_{i+1}.csv', index=False)
        
    # save metrics_on_test dictionary as csv
    metrics_on_test_df = pd.DataFrame(metrics_on_test)
    metrics_on_test_df.to_csv(f'plots/{save_folder_name}/metrics_on_test.csv', index=False)


    cm = confusion_matrix(labels_test_set, averaged_predictions_binarized, normalize='true')
    # plot confusion matrix using heatmap
    cm_analysis(labels_test_set, averaged_predictions_binarized, f'plots/{save_folder_name}/confusion_matrix.png', labels=[0,1], classes=['Non-bacilli', 'Bacilli'])
    print('Confusion matrix with predictions averaged from all models saved in plots folder')
    

    
    # save the config variable to a json file in the save_folder_name folder
    with open(f'plots/{save_folder_name}/config.json', 'w') as fp:
        json.dump(config, fp)
    print('Config file saved as json')


if __name__ == '__main__':
    main()