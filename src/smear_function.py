from src.utils import *
from src.thresholding import Thresholding
from src.postprocessing import Postprocessing
from src.cropping import Cropping
import pandas as pd
import os


def smear_pipeline(config, smear, loader):
    """
    This function is the main pipeline for the smear detection.

    :param config: the configuration file
    :param smear: the smear to be processed
    :param loader: the loader to be used
    """
    # initialize the variables
    total_number_bacilli = 0
    counter_of_blurry_images = 0
    # go through all the images in the smear
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

        # Cropping
        if postprocessing_config['crop']:
            if stats.shape[0] > 1:
                cropping_function = Cropping(img, final_image)
                cropped_images = cropping_function.crop_and_pad()
            else:
                num_bacilli = 0

        # Save the results
        labelling_dataset_config = config['labelling_dataset']
        if labelling_dataset_config['create_dataset'] and postprocessing_config['crop']:
            if stats.shape[0] > 1:

                # dataset creation
                dataframe = pd.DataFrame()
                for l in range(0, cropped_images.shape[0]):
                    d = {'image': [cropped_images[l]]}
                    df2 = pd.DataFrame(d)
                    dataframe = pd.concat([dataframe, df2], ignore_index=True)

                # save the images
                save_config = config['saving']
                if save_config['save']:
                    # save dataframe with pandas library
                    labelled_data_path = os.path.join('../labelled_data', loader.dataset_name + str(i) + '.pkl')
                    dataframe.to_pickle(labelled_data_path)

            else:
                num_bacilli = 0

        total_number_bacilli += num_bacilli
    print("Total number of bacilli: ", total_number_bacilli)
    print("Blurry images that were not considered: ", counter_of_blurry_images)

    
    