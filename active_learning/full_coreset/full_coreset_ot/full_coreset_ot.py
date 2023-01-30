import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from full_coreset_ot_train import train_coreset_cnn


from active_learning.coreset_functions import active_learning


def rescale(image):
    if np.max(image) == np.min(image):
        return image
    elif np.max(image) == 0:
        return image
    else:
        return (image - np.min(image)) / (np.max(image) - np.min(image))




def get_coreset(df, current_df, iteration):
    # reset index of the dataframe


    vectors = torch.from_numpy(rescale(df.iloc[0]['image']))
    #unsqueeze(0) to add a dimension to the tensor
    vectors = vectors.unsqueeze(0)



    for i in range(1, df.shape[0]):
        vectors = torch.cat((vectors, torch.from_numpy(rescale(df.iloc[i]['image'])).unsqueeze(0)), 0)

    acle = active_learning(vectors, 10*iteration, None)
    indexes = acle.ot_clustering()

    current_df = current_df.append(df.iloc[indexes])

    return current_df, df




# Active Learning approach using Prediction uncertanty from start to end

# query size 3 % of the dataset size 5000 ---> 150

# initialize accuracy list
accuracy_list = []


# load full dataset
df = pd.read_pickle('C://users/matteo/pycharmprojects/TBProject/labelled_data/smear_2156_17_30.pkl')
for i in range(0, 2):
    df = df.append(pd.read_pickle('C://users/matteo/pycharmprojects/TBProject/labelled_data/smear_2156_17_3' + str(i) + '.pkl'))
# update the index column
df = df.reset_index(drop=True)




current_df = pd.DataFrame(columns=['image', 'label'])



# train the model with the 150 samples
# in train_cnn save the model in a pt file named as full_pu_CNN_0.pt and in the following iterations full_pu

#5000 / 150 = 33 iterations
for i in range(0, 10):

    current_df, df = get_coreset(df, current_df, i)
    #drop last 150 samples of the current_df from df
    df = df.drop(current_df.index[-150:])
    #reset
    df = df.reset_index(drop=True)
    acc = train_coreset_cnn(current_df, i)
    accuracy_list.append(acc)

# plot the accuracy list verus number of data points
plt.plot(accuracy_list, np.arange(0, 5000, 150))
plt.show()
# save the image
plt.savefig('accuracy_list.png')