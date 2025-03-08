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
    total_number_bacilli = 0
    number_of_predicted_bacilli = 0
    for i, img in enumerate(smear):  
        print("Tile: ", i)

        # Preprocess
        preprocess_config = config['preprocessing']
        preprocessed_img = preprocess(preprocess_config, img)

        # Threshold
        threshold_config = config['thresholding']
        threshold = Thresholding(preprocessed_img, threshold_config)
        thresholded_img = threshold.apply()

        # Postprocess
        postprocessing_config = config['postprocessing']
        postprocess = Postprocessing(thresholded_img, postprocessing_config)
        whole_img_not_cleaned, final_image, num_bacilli, stats = postprocess.apply()
        # clean stats
        stats = clean_stats(stats)
        total_number_bacilli += stats.shape[0]

        # Defining the configs for the different steps
        labelling_dataset_config = config['labelling_dataset']
        save_config = config['saving']
        inference_config = config['inference']



        # Cropping
        cropped_images = "no images"
        if labelling_dataset_config['create_dataset'] or save_config['save'] or inference_config[
            'prediction'] == "CNN" or inference_config['prediction'] == "STATS":
            if stats.shape[0] > 1:
                cropping_function = Cropping(img, final_image)
                cropped_images = cropping_function.crop_and_pad()
            else:
                num_bacilli = 0


        if isinstance(cropped_images, str):
            print("No images, cannot label or save dataset or inference")
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


            if inference_config['do_inference']:
                print("Inference...")
                # do one of the possible inference
                inference = Inference(cropped_images, stats, final_image)
                if inference_config['prediction'] == 'SVM':
                    red_boxes, green_boxes = inference.svm_prediction()
                elif inference_config['prediction'] == 'CNN':
                    red_boxes, green_boxes, coordinates, predictions = inference.network_prediction()
                    number_of_predicted_bacilli += green_boxes.shape[0]
                elif inference_config['prediction'] == 'STATS':
                    red_boxes, green_boxes, coordinates = inference.ellipse_brute_prediction()


    print("Total number of supposed bacilli: ", total_number_bacilli)
    return number_of_predicted_bacilli

    
    