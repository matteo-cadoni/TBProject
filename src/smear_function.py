"""
This file contains the main pipeline for the smear detection.
The steps that are performed on every tile of the smear are:
    - Preprocessing
    - Thresholding
    - Postprocessing
    - Cropping (optional)
    - Interactive labelling (optional)
    - Dataset creation (optional)
    - Inference (optional)

We are able to count the number of objects in the image and compare it to the
number of objects that are predicted by the model to be bacilli.
"""

from src.utils import *
from src.thresholding import Thresholding
from src.postprocessing import Postprocessing
from src.cropping import Cropping
from src.interactivelabelling import InteractiveLabeling
import pandas as pd
import os
from src.inference_visualization import Inference
from n_networks.neural_net import BacilliNet
import torch
import time

def smear_pipeline(config, smear, loader):
    """This function is the main pipeline for the applying the
    computations on a smear.

    parameters
    ----------
    config: dict
        dictionary with all the parameters for the pipeline
    smear: numpy array
        image of the smear
    loader: class
        class with path to image

    returns
    -------
    total_number_bacilli: int
        total number of bacilli in the smear
    number_of_predicted_bacilli: int
        total number of bacilli predicted by the model
    """
    models = []
    for i in range(1, 6):
        path = os.path.join('src', 'saved_models', "cnn_results", 'model_' + str(i) + '.pth')
        model_i = BacilliNet()
        model_i.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        models.append(model_i)        # get the dataset, with dataset loader
    
    
    total_number_bacilli = 0
    number_of_predicted_bacilli = 0
    tiles_bacilli = {}
    for i, img in enumerate(smear):  
        print("Tile", i,"/", smear.shape[0])

        # Preprocess
        start_time_preprocess = time.time()
        preprocess_config = config['preprocessing']
        preprocessed_img = preprocess(preprocess_config, img)
        end_time_preprocess = time.time()
        time_preprocess = end_time_preprocess - start_time_preprocess

        # Threshold
        start_time_threshold = time.time()
        threshold_config = config['thresholding']
        threshold = Thresholding(preprocessed_img, threshold_config)
        thresholded_img = threshold.apply()
        end_time_threshold = time.time()
        time_threshold = end_time_threshold - start_time_threshold

        # Postprocess
        start_time_postprocess = time.time()
        postprocessing_config = config['postprocessing']
        postprocess = Postprocessing(thresholded_img, postprocessing_config)
        whole_img_not_cleaned, final_image, num_bacilli, stats = postprocess.apply()
        # clean stats
        stats = clean_stats(stats)
        total_number_bacilli += stats.shape[0]
        end_time_postprocess = time.time()
        time_postprocess = end_time_postprocess - start_time_postprocess

        # Defining the configs for the different steps
        labelling_dataset_config = config['labelling_dataset']
        save_config = config['saving']
        inference_config = config['inference']



        # Cropping
        start_time_cropping = time.time()
        cropped_images = "no images"
        if labelling_dataset_config['create_dataset'] or save_config['save'] or inference_config[
            'prediction'] == "CNN" or inference_config['prediction'] == "STATS":
            if stats.shape[0] > 1:
                cropping_function = Cropping(img, final_image)
                cropped_images = cropping_function.crop_and_pad()
            else:
                num_bacilli = 0
        end_time_cropping = time.time()
        time_cropping = end_time_cropping - start_time_cropping

        if isinstance(cropped_images, str):
            print("No images, cannot label or save dataset or inference. \n")
            tiles_bacilli[i] = 0
            
        else:
            # Save the results
            labelling_dataset_config = config['labelling_dataset']
            if labelling_dataset_config['create_dataset'] and postprocessing_config['crop']:
                if stats.shape[0] > 1:
                    i_l = InteractiveLabeling(cropped_images)
                    labels = i_l.run()

                    # dataset creation
                    dataframe = pd.DataFrame()
                    for l in range(0, cropped_images.shape[0]):
                        # le stats coincidono con le immagini ?
                        d = {'image': [cropped_images[l]], 'label': [labels[l]], 'stats': [stats[l]]}
                        df2 = pd.DataFrame(d)
                        dataframe = pd.concat([dataframe, df2], ignore_index=True)

                    # save the images

                    if save_config['save']:
                        # save dataframe with pandas library
                        labelled_data_path = os.path.join('labelled_data', loader.dataset_name + str(i) + '.pkl')
                        dataframe.to_pickle(labelled_data_path)
                        print("Dataset saved in: " + labelled_data_path)
                    if save_config['save_stats']:
                        # create dataframe with stats for each sample then save it as a .pkl file
                        stats_dataframe = pd.DataFrame(stats)
                        stats_dataframe_path = os.path.join('labelled_data', 'stats_' + loader.dataset_name + str(i) + '.pkl')
                        stats_dataframe.to_pickle(stats_dataframe_path)
                        print("Stats saved in: " + stats_dataframe_path)

                else:
                    num_bacilli = 0

            start_time_inference = time.time()
            if inference_config['do_inference']:
                #print("Inference...")
                # do one of the possible inference
                inference = Inference(cropped_images, stats, final_image)
                if inference_config['prediction'] == 'SVM':
                    red_boxes, green_boxes = inference.svm_prediction()
                elif inference_config['prediction'] == 'CNN':
                    red_boxes, green_boxes, predictions = inference.network_prediction(models)
                    new_bacilli = green_boxes.shape[0]
                    assert new_bacilli == np.sum(predictions), "The number of bacilli predicted by the model is different from the number of bacilli predicted by the postprocessing"
                    print("Number of bacilli predicted by the model for this tile: ", new_bacilli)
                    tiles_bacilli[i] = new_bacilli
                    number_of_predicted_bacilli += green_boxes.shape[0]
                    print("New total number of bacilli in the whole smear: ", number_of_predicted_bacilli, "\n")
                elif inference_config['prediction'] == 'STATS':
                    red_boxes, green_boxes, coordinates = inference.ellipse_brute_prediction()
            end_time_inference = time.time()
            time_inference = end_time_inference - start_time_inference


    
        print("Time of preprocessing: ", time_preprocess)
        print("Time of thresholding: ", time_threshold)
        print("Time of postprocessing: ", time_postprocess)
        print("Time of cropping: ", time_cropping)
        if isinstance(cropped_images, str) == False:
            print("Time of inference: ", time_inference)
    print("Total number of supposed bacilli: ", total_number_bacilli)    
    
    return number_of_predicted_bacilli, tiles_bacilli, total_number_bacilli

    
    