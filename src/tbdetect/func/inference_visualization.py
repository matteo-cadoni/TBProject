import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tbdetect.neural_net import BacilliNet, ChatGPT
import pandas as pd
import joblib


class Inference:
    """ Class to predict the class of the bacilli in the image.
    Attributes:
    ----------
    cropped_images: list
        list of the cropped bacilli images
    stats: list
        list of the stats of the bacilli
    final_image: numpy array
        masked image
    Methods:
    -------
    get_dataset()
        Get the dataset to do inference with the CNN.
    get_boxes(predictions)
        Get the boxes to draw in napari, green for bacilli, red for non-bacilli.
    network_prediction()
        Predict the class of the images, using pretrained neural network.
    stats_prediction()
        Predict the class of the images, using the stats.
    ellipse_brute_prediction()
        Predict the class of the images, using the contours and finding enclosing ellipse.
    svm_prediction()
        Predict the class of the images, using a on the stats pretrained SVM.
    get_hu_moments()
        Get elongation Hu-moment for every object in the image.
    """
    def __init__(self, cropped_images, stats, final_image):
        """
        parameters:
        ----------
        cropped_images: list
            list of the cropped bacilli images
        stats: list
            list of the stats of the bacilli
        final_image: numpy array
            masked image
        """
        self.final_image = final_image
        self.cropped_images = cropped_images
        self.stats = stats
        # load the model
        self.PATH = os.path.join(os.path.dirname(__file__), 'n_networks', 'model.pth')
        # initialize the model
        self.model = ChatGPT()
        # set parameters
        self.model.load_state_dict(torch.load(self.PATH))
        self.models = []
        for i in range(1, 6):
            path = os.path.join(os.path.dirname(__file__), 'n_networks', 'model_' + str(i) + '.pth')
            model_i = BacilliNet()
            model_i.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.models.append(model_i)
        
        # get the dataset, with dataset loader
        dataset = self.get_dataset()
        inference_dataset = MyDataset(dataset)
        self.inference_dataset_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    def network_prediction(self):
        """ 
        Predict the class of the images, using the neural network,
        return arrays needed to draw the boxes in napari.
        Returns:
        -------
        red_boxes: list
            list of the boxes to draw in napari, red for non-bacilli
        green_boxes: list
            list of the boxes to draw in napari, green for bacilli
        """
        # initialize prediction array
        predictions = np.array([])
        # coordinates taken from ellipse approach
        _, _, coordinates = self.ellipse_brute_prediction()
        # iterate over the dataset and predict the class
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.inference_dataset_loader):
                image = data
                image = image.to(torch.float32)
                image = image.view(1, 1, 50, 50)
                # output = self.model(image)
                outputs = []
                for i in range(5):
                    outputs.append(self.models[i](image))
    
                output = torch.mean(torch.stack(outputs), dim=0)
                output = output.squeeze(1)
                if output > 0.5:
                    predictions = np.append(predictions, 1)
                else:
                    predictions = np.append(predictions, 0)
        # use get_boxes to get the boxes
        red_boxes, green_boxes = self.get_boxes(predictions)
        return  red_boxes, green_boxes, coordinates, predictions

    def ellipse_brute_prediction(self):
        """ Find the contours of the bacilli in the image.
        And find the enclosing ellipse for every bacilli.
        returns
        -------
        red_boxes: list
            list of the boxes to draw in napari, red for non-bacilli
        green_boxes: list
            list of the boxes to draw in napari, green for bacilli
        """
        predictions = np.array([])
        axes_coordinates = np.empty((0, 2))
        for i in range(0, self.stats.shape[0]):
            fake_contours = np.zeros((5, 1, 2), dtype=np.int32)
            contours, _ = cv2.findContours(self.final_image[self.stats[i][1]:self.stats[i][1]+self.stats[i][3],
                            self.stats[i][0]:self.stats[i][0]+self.stats[i][2]], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) < 5: 
                fake_contours[0:len(cnt),:,:] = cnt
                cnt = fake_contours
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (ma, MA), angle = ellipse
            axes_coordinates = np.append(axes_coordinates, [[MA, ma]], axis=0)
            if MA/ma > 1.5: #  MA/ma > 2
                predictions = np.append(predictions, 1)
            else: # red boxes
                predictions = np.append(predictions, 0)

            if self.stats[i][4] > 200:
                predictions[i] = 0   

        red_boxes, green_boxes = self.get_boxes(predictions)
        return red_boxes, green_boxes, axes_coordinates, predictions

    def svm_prediction(self):
        """ Predict the class of the images, using a svm that was trained on the stats.
        returns
        -------
        red_boxes: list
            list of the boxes to draw in napari, red for non-bacilli
        green_boxes: list
            list of the boxes to draw in napari, green for bacilli
        """
        # drop coordinates from stats
        self.stats = self.stats[:, 2:]
        # create a stats dataframe
        df = pd.DataFrame(self.stats)
        # load the svm model
        loaded_model = joblib.load('svm_results/svm.pkl')
        # predict the class
        predictions = loaded_model.predict(df)
        return self.get_boxes(predictions)

    def get_boxes(self, predictions):
        """ Get the boxes to draw in napari, based on the predictions.
        parameters
        ----------
        predictions: numpy array
            array of predictions
        returns
        -------
        red_boxes: list
            list of the boxes to draw in napari, red for non-bacilli
        green_boxes: list
            list of the boxes to draw in napari, green for bacilli
        """

        red_boxes = np.array([[0, 0], [0, 0]])
        red_boxes = np.stack((red_boxes, red_boxes), axis=0)
        green_boxes = np.array([[0, 0], [0, 0]])
        green_boxes = np.stack((green_boxes, green_boxes), axis=0)
        # create a rectangle array
        for i in range(0, self.stats.shape[0]-1):
            y = self.stats[i+1][0] - 5
            x = self.stats[i+1][1] - 5
            w = self.stats[i+1][3]
            h = self.stats[i+1][2]
            if predictions[i] == 0:
                red_boxes = np.concatenate((red_boxes, [np.array([[x, y], [x + w + 10, y + h + 10]])]), axis=0)

            else:
                green_boxes = np.concatenate((green_boxes, [np.array([[x, y], [x + w + 10, y + h + 10]])]), axis=0)
        # remove first two by two matrix that where added to initialize the array
        red_boxes = np.delete(red_boxes, 0, axis=0)
        red_boxes = np.delete(red_boxes, 0, axis=0)
        green_boxes = np.delete(green_boxes, 0, axis=0)
        green_boxes = np.delete(green_boxes, 0, axis=0)
        return red_boxes, green_boxes

    def get_dataset(self):
        """Create a dataset from the cropped images, to be used in the neural network.
        returns
        -------
        dataframe: pandas dataframe
            dataframe with the cropped images
        """
        dataframe = pd.DataFrame()
        for i in range(0, self.cropped_images.shape[0]):
            d = {'image': [self.cropped_images[i]]}
            df2 = pd.DataFrame(d)
            dataframe = pd.concat([dataframe, df2], ignore_index=True)
        return dataframe
    
class MyDataset(Dataset):
    """ Dataset class for the neural network.
    parameters
    ----------
    data: pandas dataframe
        dataframe with the cropped images
    returns
    -------
    torch tensor:
        tensor with the cropped images
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data.iloc[index]['image']
        #img = np.array(img)
        # change image values to be in 0,1 range instead of 0, 16000
        # if img.max() > 0:
        #     img = img / img.max()
        # convert to uint4
        if (np.max(img) - np.min(img) != 0):
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) - 0.5
        # img = img / 16000
        return torch.tensor(img, dtype=torch.float32)











