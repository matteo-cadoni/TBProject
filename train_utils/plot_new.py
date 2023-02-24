from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


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

    # plot
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="o")
    plt.xlabel("Epoch", fontsize=12)
    # make xlabel text bigger
    plt.xticks(fontsize=12)
    # make ylabel text bigger

    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Validation Accuracy over Epochs", fontsize=15)
    # add legend

    # put legend in the bottom right part of the plot

    plt.show()
    # interpolate between points
    plt.plot(x, y, "o-", color="blue")
    plt.fill_between(x, min_acc, max_acc, alpha=0.2)
    plt.legend(
        ["Average Validation Accuracy", "Min and Max Validation Accuracy"],
        fontsize=12,
        loc="lower right",
    )

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
    yerr_train = [
        avg_training_loss - min_training_loss,
        max_training_loss - avg_training_loss,
    ]
    y_val = avg_val_loss
    yerr_val = [avg_val_loss - min_val_loss, max_val_loss - avg_val_loss]

    # plot training loss
    fig = plt.figure()
    plt.errorbar(x, y_train, yerr=yerr_train, fmt="o")
    plt.errorbar(x, y_val, yerr=yerr_val, fmt="o")
    # interpolate between points
    plt.plot(x, y_train, "o-", color="blue")
    plt.fill_between(x, min_training_loss, max_training_loss, alpha=0.2)
    # change background color to white
    plt.rcParams["axes.facecolor"] = "white"
    plt.plot(x, y_val, "o-", color="orange")
    plt.fill_between(x, min_val_loss, max_val_loss, alpha=0.2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title("Training/Test Loss", fontsize=15)
    plt.legend(
        [
            "Average Training Loss",
            "Min and Max Train Loss",
            "Average Validation Loss",
            "Min and Max Validation Loss",
        ],
        fontsize=12,
        loc="upper right",
    )
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


def cm_analysis(y_true, y_pred, filename, labels, classes, ymap=None, figsize=(17, 17)):
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
            # if i == j:
            s = cm_sum[i]
            annot[i, j] = "%.2f%%\n%d/%d" % (p, c, s)
            # elif c == 0:
            #    annot[i, j] = ''
            # else:
            #    annot[i, j] = '%.2f%%\n%d' % (p, c)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = "True Label"
    cm.columns.name = "Predicted Label"
    fig, ax = plt.subplots(figsize=figsize)
    plt.yticks(va="center")

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        ax=ax,
        xticklabels=classes,
        cbar=True,
        cbar_kws={"format": PercentFormatter()},
        yticklabels=classes,
        cmap="Reds",
    )
    plt.savefig(filename, bbox_inches="tight")
    # close figure
    plt.close(fig)
