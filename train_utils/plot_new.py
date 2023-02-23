from matplotlib import pyplot as plt
import numpy as np

def plot_validation_accuracies(val_acc, save_folder_name):
    accuracies_matrix = np.empty((0, len(val_acc[0])))
    for acc in val_acc:
        acc = np.array(acc)
        accuracies_matrix = np.vstack((accuracies_matrix, acc))
        
    # average accuracy over columns
    avg_acc = np.mean(accuracies_matrix, axis=0)
    assert len(avg_acc) == len(val_acc[0])
    
    # max accuracy over columns
    max_acc = np.max(accuracies_matrix, axis=0)
    assert len(max_acc) == len(val_acc[0])
    
    # min accuracy over columns
    min_acc = np.min(accuracies_matrix, axis=0)
    assert len(min_acc) == len(val_acc[0])
    
    x = np.arange(len(val_acc[0]))
    y = avg_acc
    yerr = [avg_acc - min_acc, max_acc - avg_acc]
    
    #plot 
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='o')
    plt.xlabel('Epoch', fontsize=12)
    # make xlabel text bigger 
    plt.xticks(fontsize=12)
    # make ylabel text bigger

    
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Validation Accuracy over Epochs', fontsize=15)
    # add legend
    
    # put legend in the bottom right part of the plot

    plt.show()
    # interpolate between points
    plt.plot(x, y, 'o-', color='blue')
    plt.fill_between(x, min_acc, max_acc, alpha=0.2)
    plt.legend(['Average Validation Accuracy', 'Min and Max Validation Accuracy'], fontsize=12, loc='lower right')
    
    # save plot to file
    plt.savefig(f"plots/{save_folder_name}/validation_accuracies.png")
    
    # close plot 
    plt.close()
    
def plot_training_and_validation_losses(training_losses, val_losses, save_folder_name):
    training_losses_matrix = np.empty((0, len(training_losses[0])))
    for loss in training_losses:
        loss = np.array(loss)
        training_losses_matrix = np.vstack((training_losses_matrix, loss))
        
    validation_losses_matrix = np.empty((0, len(val_losses[0])))
    for loss in val_losses:
        loss = np.array(loss)
        validation_losses_matrix = np.vstack((validation_losses_matrix, loss))
    
    # average loss over columns
    avg_training_loss = np.mean(training_losses_matrix, axis=0)
    assert len(avg_training_loss) == len(training_losses[0])
    
    # max loss over columns
    max_training_loss = np.max(training_losses_matrix, axis=0)
    assert len(max_training_loss) == len(training_losses[0])
    
    # min loss over columns
    min_training_loss = np.min(training_losses_matrix, axis=0)
    
    # average loss over columns
    avg_val_loss = np.mean(validation_losses_matrix, axis=0)
    assert len(avg_val_loss) == len(val_losses[0])
    
    # max loss over columns
    max_val_loss = np.max(validation_losses_matrix, axis=0)
    assert len(max_val_loss) == len(val_losses[0])
    
    # min loss over columns
    min_val_loss = np.min(validation_losses_matrix, axis=0)
    
    x = np.arange(len(training_losses[0]))
    y_train = avg_training_loss
    yerr_train = [avg_training_loss - min_training_loss, max_training_loss - avg_training_loss]
    y_val = avg_val_loss
    yerr_val = [avg_val_loss - min_val_loss, max_val_loss - avg_val_loss]
    
    # plot training loss
    fig = plt.figure()
    plt.errorbar(x, y_train, yerr=yerr_train, fmt='o')
    plt.errorbar(x, y_val, yerr=yerr_val, fmt='o')
    # interpolate between points
    plt.plot(x, y_train, 'o-', color='blue')
    plt.fill_between(x, min_training_loss, max_training_loss, alpha=0.2)
    # change background color to white 
    plt.rcParams['axes.facecolor'] = 'white'
    plt.plot(x, y_val, 'o-', color='orange')
    plt.fill_between(x, min_val_loss, max_val_loss, alpha=0.2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training/Test Loss', fontsize=15)
    plt.legend(['Average Training Loss', 'Min and Max Train Loss', 'Average Validation Loss', 'Min and Max Validation Loss'], fontsize=12, loc='upper right')
    # put legend in the bottom right part of the plot
    # make xlabel text bigger 
    plt.xticks(fontsize=12)
    # make ylabel text bigger
    plt.yticks(fontsize=12)
    
    plt.show()
    
    # save plot to file
    fig.savefig(f"plots/{save_folder_name}/losses_cross_validation.png")
    
    # close plot
    plt.close()