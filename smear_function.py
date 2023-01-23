from src.preprocess import Preprocessing
from utils import *
from src.thresholding import Thresholding
from src.postprocessing import Postprocessing
from src.visualization import visualize_all_list_napari, add_bounding_boxes, is_blurry_laplacian
from cropping import Cropping
import pandas as pd
import os



    
def smear_pipeline(config, smear, loader):
    total_number_bacilli = 0
    counter_of_blurry_images = 0
    for i, img in enumerate(smear):  
        print("Tile: ", i)
        if config['load']['blurry_deselection']:
            if is_blurry_laplacian(img):
                print("It's blurry")
                # counter of the blurry images:
                counter_of_blurry_images += 1
                pass
            else:
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

                if postprocessing_config['crop']:

                    ######## BEGIN CROPPING ########
                    cropping_function = Cropping(img, final_image)
                    cropped_images = cropping_function.crop_and_pad()
                    ######## END CROPPING ########

                labelling_dataset_config = config['labelling_dataset']
                if labelling_dataset_config['create_dataset']:



                    ######## BEGIN DATASET CREATION ########
                    dataframe = pd.DataFrame()
                    for i in range(0, cropped_images.shape[0]):
                        d = {'image': [cropped_images[i]]}
                        df2 = pd.DataFrame(d)
                        dataframe = pd.concat([dataframe, df2], ignore_index=True)
                    ######## END DATASET CREATION ########

                    ######## BEGIN SAVING ########
                    save_config = config['saving']
                    if save_config['save']:
                        # save dataframe with pandas library
                        labelled_data_path = os.path.join('labelled_data', loader.dataset_name + '.pkl')
                        dataframe.to_pickle(labelled_data_path)
                    ######## END SAVING ########
                total_number_bacilli += num_bacilli
                if total_number_bacilli > 100000:
                    break
        else:
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
            print(num_bacilli)
            if i > 621:
                if postprocessing_config['crop']:
                    if stats.shape[0]>1 and num_bacilli<4000:
                        ######## BEGIN CROPPING ########
                        cropping_function = Cropping(img, final_image)
                        cropped_images = cropping_function.crop_and_pad()
                        ######## END CROPPING ########
                    else:
                        pass

                labelling_dataset_config = config['labelling_dataset']
                if labelling_dataset_config['create_dataset']:
                    if stats.shape[0] > 1 and num_bacilli < 4000:

                        ######## BEGIN DATASET CREATION ########
                        dataframe = pd.DataFrame()
                        for i in range(0, cropped_images.shape[0]):
                            d = {'image': [cropped_images[i]]}
                            df2 = pd.DataFrame(d)
                            dataframe = pd.concat([dataframe, df2], ignore_index=True)
                        ######## END DATASET CREATION ########

                        ######## BEGIN SAVING ########
                        save_config = config['saving']
                        if save_config['save']:
                            # save dataframe with pandas library
                            labelled_data_path = os.path.join('labelled_data', loader.dataset_name+ str(i) + '.pkl')
                            dataframe.to_pickle(labelled_data_path)
                        ######## END SAVING ########
                    else:
                        num_bacilli = 0


            total_number_bacilli += num_bacilli
            if i == 700:
                break
    print("Total number of bacilli: ", total_number_bacilli)  
    print("Blurry images that were not considered: ", counter_of_blurry_images)
    
    