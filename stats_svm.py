import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


class Stats_svm():
    def __init__(self, dataset, stats):
        self.dataset = dataset
        self.stats = stats

        self.number_of_entries = self.dataset.shape[0]


    def create_dataframe(self):
        #df = pd.DataFrame(self.dataset)
        df = self.dataset
        #delete the 'image' column
        df = df.drop(['image'], axis=1)
        #add a column to the dataset for each column in stats
        df['area'] = self.stats[0:self.number_of_entries,4]
        df['hight'] = self.stats[0:self.number_of_entries,2]
        df['width'] = self.stats[0:self.number_of_entries,3]
        return df
    def train_svm(self):
        df = self.create_dataframe()
        #create a dataframe with the labels
        labels = df['label']
        #drop the labels from the dataset
        df = df.drop('label', axis=1)
        #split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=0)
        #initialize the svm
        self.clf = svm.SVC(kernel='linear', C=1)
        #train the svm
        self.clf.fit(X_train, y_train)
        #test the svm
        self.clf.score(X_test, y_test)
        #save the svm
        joblib.dump(self.clf, 'svm_results/svm.pkl')
        # visaualize the svm
    
    def visualize_svm(self):
        #load the svm
        self.clf = joblib.load('svm_results/svm.pkl')
        #create a dataframe
        df = self.create_dataframe()
        #create a dataframe with the labels
        labels = df['label']
        #drop the labels from the dataset
        df = df.drop('label', axis=1)
        #split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=0)
        #test the svm
        y_pred = self.clf.predict(X_test)
        #print the accuracy
        print("Accuracy:", accuracy_score(y_test, y_pred))


# load .pkl file from labelled_data folder
print("Loading stats")
stats = pd.read_pickle('labelled_data/stats_tile_673_smear_2156_17_3.pkl')
# turn stats dataframe into a numpy array
stats = stats.to_numpy()
# remove background stats, remove first row
stats = np.delete(stats, 0, 0)
print("Stats loaded into numpy array!")

data = pd.read_pickle('labelled_data/tile_673_smear_2156_17_3.pkl')
print("Dataset loaded")

procedure = Stats_svm(data, stats)

#train the svm
procedure.train_svm() 

procedure.visualize_svm()