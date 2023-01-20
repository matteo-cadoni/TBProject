import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from neural_net import ChatGPT
import pandas as pd
import joblib

class Inference():
    def __init__(self, cropped_images, stats):
        self.cropped_images = cropped_images
        self.stats = stats
        self.PATH = 'n_networks/model.pth'
        #load model from torch file, pth file
        self.model = ChatGPT()
        self.model.load_state_dict(torch.load(self.PATH))
        dataset = self.get_dataset()
        inference_dataset = MyDataset(dataset)
        self.inference_dataset_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)


    def network_prediction(self):
        predictions = np.array([])
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.inference_dataset_loader):
                image = data
                image = image.to(torch.float32)
                image = image.view(1, 1, 50, 50)
                output = self.model(image)
                output = output.squeeze(1)
                if output > 0.5:
                    predictions = np.append(predictions, 1)
                else:
                    predictions = np.append(predictions, 0)
        return self.get_boxes(predictions)


    def stats_prediction(self):
        predictions = np.array([])
        for i in range(0, self.stats.shape[0]):
            # get vertically lying bacilli
            if 4*self.stats[i][2] > self.stats[i][3]:
                predictions = np.append(predictions, 1)
            # get horizontally lying bacilli
            elif 4*self.stats[i][3] > self.stats[i][2]:
                predictions = np.append(predictions, 1)
            #get vertical bacilli
            elif abs(self.stats[i][2] -self.stats[i][3]) < 4 and self.stats[i][4]< 125:
                predictions = np.append(predictions, 1)
            else:
                predictions = np.append(predictions, 0)
        return self.get_boxes(predictions)

    def svm_prediction(self):
        #load the svm model
        loaded_model = joblib.load('svm.pkl')
        predictions = loaded_model.predict(self.cropped_images)
        return self.get_boxes(predictions)



    def get_boxes(self, predictions):

        red_boxes = np.array([[0,0],[0,0]])
        red_boxes = np.stack((red_boxes, red_boxes), axis=0)
        green_boxes = np.array([[0,0],[0,0]])
        green_boxes = np.stack((green_boxes, green_boxes), axis=0)
        #create a rectangle array

        for i in range(0, predictions.shape[0]):
            y = self.stats[i+1][0] - 5
            # x_max = coordinates[i][0]
            x = self.stats[i+1][1] - 5
            # y_max = coordinates[i][1]
            w = self.stats[i+1][3]
            h = self.stats[i+1][2]
            if predictions[i] == 0:
                red_boxes = np.concatenate((red_boxes, [np.array([[x, y], [x + w + 10, y + h + 10]])]), axis=0)

            else:
                green_boxes = np.concatenate((green_boxes, [np.array([[x, y], [x + w + 10, y + h + 10]])]), axis=0)
        #remove first two by two matrix
        red_boxes = np.delete(red_boxes, 0, axis=0)
        red_boxes = np.delete(red_boxes, 0, axis=0)
        green_boxes = np.delete(green_boxes, 0, axis=0)
        green_boxes = np.delete(green_boxes, 0, axis=0)

        return red_boxes, green_boxes


    def get_dataset(self):
        dataframe = pd.DataFrame()
        for i in range(0, self.cropped_images.shape[0]):
            d = {'image': [self.cropped_images[i]]}
            df2 = pd.DataFrame(d)
            dataframe = pd.concat([dataframe, df2], ignore_index=True)
        return dataframe


# dataloaders for train and test
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data.iloc[index]['image']
        # change image values to be in 0,1 range instead of 0, 16000
        img = img / 16000

        return torch.tensor(img, dtype=torch.float32)











