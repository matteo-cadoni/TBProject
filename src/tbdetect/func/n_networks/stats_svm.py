import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import joblib
import os

class StatsSvm:
    """
    train the svm to recognize bacilli based on the stats
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.stats = np.array(self.dataset['stats'].tolist())
        self.number_of_entries = self.dataset.shape[0]

    def create_dataframe(self):
        # df = pd.DataFrame(self.dataset)
        df = self.dataset
        # we do not need the 'image' column
        df = df.drop(['image'], axis=1)
        if 'predictions_ellipse' in df.columns:
            # we do not need the 'predictions_ellipse' column
            df = df.drop(['predictions_ellipse'], axis=1)
        # we separate each of the stats in the stat column into different columns
        df['area'] = self.stats[0:self.number_of_entries, 4]
        df['hight'] = self.stats[0:self.number_of_entries, 2]
        df['width'] = self.stats[0:self.number_of_entries, 3]
        #df['sqrt(height^2+width^2)'] = np.sqrt((self.stats[0:self.number_of_entries, 2])**2+(self.stats[0:self.number_of_entries, 3])**2)
       
        # we do not need the 'stats' column anymore
        df = df.drop(['stats'], axis=1)
        return df

    def train_svm(self):
        df = self.create_dataframe()
        # create a dataframe with the labels
        labels = df['label']
        # drop the labels from the dataset
        df = df.drop('label', axis=1)
        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=0)
        # initialize the svm
        self.clf = svm.SVC(kernel='sigmoid', C=1)
        # train the svm
        self.clf.fit(X_train, y_train)
        # test the svm
        self.clf.score(X_test, y_test)
        # save the svm
        joblib.dump(self.clf, 'svm_results/svm.pkl')

    def tests_for_svm(self):
        # load the svm
        self.clf = joblib.load('svm_results/svm.pkl')
        # create a dataframe
        df = self.create_dataframe()
        # create a dataframe with the labels
        labels = df['label']
        # drop the labels from the dataset
        df = df.drop('label', axis=1)
        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=0)
        # test the svm
        y_pred = self.clf.predict(X_test)
        # print the accuracy
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Recall", recall_score(y_test, y_pred))
        print("Precision", precision_score(y_test, y_pred))
        print("F1 score", f1_score(y_test, y_pred))
        print("Confusion matrix", confusion_matrix(y_test, y_pred))
        print("Number of bacilli predicted: ", y_pred.sum())
        print("Number of bacilli ground truth: ", y_test.sum())

    def visualize_3D_scatterplot_for_stats(self, df):
        # create a 3D scatterplot for the stats
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['area'], df['hight'], df['width'], c=df['label'], cmap='viridis', marker='o', label='bacilli')
        ax.set_xlabel('area')
        ax.set_ylabel('height')
        ax.set_zlabel('width')
        plt.legend()
        plt.show()

    def visualize_2D_scatterplot_for_stats(self, df):
        # create a 2D scatterplot for the stats
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(df['area'], df['sqrt(height^2+width^2)'], c=df['label'], marker='o', cmap='jet', label='bacilli')
        ax.set_xlabel('Area', fontsize=14)
        ax.set_ylabel('sqrt(Height^2+Width^2)', fontsize=14)
        # increase font size of the labels and axis
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(['Bacilli', 'No bacilli'], loc='upper right', fontsize=14)
        plt.show()
    
whole_dataset = pd.DataFrame()
for file in os.listdir('dataframe'):
    df = pd.read_pickle('dataframe/' + file)
    whole_dataset = pd.concat([whole_dataset, df], ignore_index=True)

other_dataset = pd.read_pickle('dataframe/data_with_stats.pkl')
    
print("Dataset created, size: ", other_dataset.shape)

procedure = StatsSvm(other_dataset)
# train the svm
procedure.train_svm()

procedure.tests_for_svm()

procedure.visualize_3D_scatterplot_for_stats(procedure.create_dataframe())

#procedure.visualize_2D_scatterplot_for_stats(procedure.create_dataframe())
