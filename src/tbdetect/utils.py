from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
import warnings

try:
    import napari
except ImportError:
    warnings.warn("The package napari is not installed. Install napari to enable visualization features. Attention: installation of napari differs from machine to machine. Make sure to choose the right installation command for your machine.")
    _has_napari = False
else:
    _has_napari = True


from tbdetect.func.loader import Loader
from tbdetect.func.preprocess import Preprocessing
from tbdetect.func.thresholding import Thresholding
from tbdetect.func.postprocessing import Postprocessing
from tbdetect.func.cropping import Cropping
from tbdetect.func.interactivelabelling import InteractiveLabeling
from tbdetect.func.interactive_config import InteractiveConfig, change_yaml
from tbdetect.func.inference_visualization import Inference
from tbdetect.func.visualization import visualize_all_list_napari, add_bounding_boxes

def interactive_config(config):  
    while True:
        i_c = InteractiveConfig()
        configurations = i_c.run()
        if i_c.error:
            continue
        else:
            break
    return change_yaml(configurations, config)

def load(load_config):
    loader = Loader(load_config['czi_path'], load_config['tile'])
    loader.load()
    img = loader.data_array
    if load_config['tile'] == 'None':
        print(f"Smear succesfully loaded, shape: {img.shape}")
    else: 
        print(f"Tile succesfully loaded, shape: {img.shape}")
    return img, loader

def preprocess(preprocess_config, tile):
    preprocess = Preprocessing(tile)
    if preprocess_config['algorithm'] == "otsu" or preprocess_config['algorithm'] == "hard":
        return preprocess.sharpen()
    else:
        return preprocess.rescale()
    
def clean_stats(stats):
    """
    Delete connected components that are too small, and
    connected components that are too large.
    """
    # make a copy of stats
    stats1 = stats.copy()
    #indices to delete
    indices = []
    # delete
    for i in range(0, stats.shape[0]):
        if stats[i, 4] > 625:
            # append index
            indices.append(i)

        if stats[i, 4] < 20:
            indices.append(i)
    # delete
    stats1 = np.delete(stats1, indices, axis=0)
    return stats1

def smear_pipeline(config, smear, loader):
    """
    This function is the main pipeline for the smear detection.
    :param config: the configuration file
    :param smear: the smear to be processed
    :param loader: the loader to be used
    """
    # initialize the variables
    total_number_bacilli = 0
    # go through all the images in the smear
    for i, img in enumerate(smear):  
        print("Tile: ", i)

        # Preprocess
        preprocess_config = config['thresholding']
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
        # Cropping
        cropped_images = "no images"
        if postprocessing_config['crop']:
            if stats.shape[0] > 1:
                cropping_function = Cropping(img, final_image)
                cropped_images = cropping_function.crop_and_pad()
            else:
                num_bacilli = 0

        save_config = config['saving']
        if isinstance(cropped_images, str):
            print("No images, cannot label or save dataset or do inference")
        else:
            # Save the results
            labelling_dataset_config = config['labelling_dataset']
            if labelling_dataset_config['create_dataset'] and postprocessing_config['crop']:
                if stats.shape[0] > 1 and i > 228:
                    i_l = InteractiveLabeling(cropped_images)
                    labels = i_l.run()

                    # dataset creation
                    dataframe = pd.DataFrame()
                    for l in range(0, cropped_images.shape[0]):
                        d = {'image': [cropped_images[l]], 'label': [labels[l]]}
                        df2 = pd.DataFrame(d)
                        dataframe = pd.concat([dataframe, df2], ignore_index=True)

                    # save the images

                    if save_config['save']:
                        # save dataframe with pandas library
                        labelled_data_path = os.path.join('D:/dataframe', loader.dataset_name + str(i) + '.pkl')
                        dataframe.to_pickle(labelled_data_path)

                else:
                    num_bacilli = 0
        if save_config['save_stats']:
            # create dataframe with stats for each sample then save it as a .pkl file
            stats_dataframe = pd.DataFrame(stats)
            stats_dataframe_path = os.path.join('labelled_data', 'stats_' + loader.dataset_name + "_" + str(i) + '.pkl')
            stats_dataframe.to_pickle(stats_dataframe_path)
            print("Stats saved in: " + stats_dataframe_path)
        total_number_bacilli += num_bacilli
    print("Total number of bacilli: ", total_number_bacilli)
    
    
def tile_pipeline(config, img, loader):
    # Preprocess
    preprocess_config = config['thresholding']
    preprocessed_img = preprocess(preprocess_config, img)
    
    
    ######## BEGIN THRESHOLDING ########
    threshold_config = config['thresholding']
    threshold = Thresholding(preprocessed_img, threshold_config)
    thresholded_img = threshold.apply()
    ######## END THRESHOLDING ########
    
    ######## BEGIN POSTPROCESSING ########
    postprocessing_config = config['postprocessing']
    postprocess = Postprocessing(thresholded_img, postprocessing_config)
    whole_img_not_cleaned, final_image, num_bacilli, stats = postprocess.apply()
    stats = clean_stats(stats)
    ######## END POSTPROCESSING ########

    ######## BEGIN ADD BOUNDING BOXES ########
    image_boxes = add_bounding_boxes(img, stats)
    ######## END BOUNDING BOXES ########

    if postprocessing_config['crop']:
        ######## BEGIN CROPPING ########
        cropping_function = Cropping(img, stats)
        cropped_images = cropping_function.crop_and_pad()
        ######## END CROPPING ########

    labelling_dataset_config = config['labelling_dataset']
    if labelling_dataset_config['create_dataset']:

        ######## BEGIN INTERACTIVE LABELING #######
        i_l = InteractiveLabeling(cropped_images)
        labels = i_l.run()
        ######### END INTERACTIVE LABELING ########

        ######## BEGIN DATASET CREATION ########
        dataframe = pd.DataFrame()
        for i in range(0, labels.shape[0]):
            d = {'image': [cropped_images[i]], 'label': [labels[i]]}
            df2 = pd.DataFrame(d)
            dataframe = pd.concat([dataframe, df2], ignore_index=True)
        ######## END DATASET CREATION ########

    ######## BEGIN SAVING ########
    save_config = config['saving']
    if save_config['save']:
        # save dataframe with pandas library
        labelled_data_path = os.path.join('labelled_data', loader.dataset_name + '.pkl')
        dataframe.to_pickle(labelled_data_path)
        print("Dataset saved in: " + labelled_data_path)
    if save_config['save_stats']:
        # create dataframe with stats for each sample then save it as a .pkl file
        stats_dataframe = pd.DataFrame(stats)
        stats_dataframe_path = os.path.join('labelled_data', 'stats_' + loader.dataset_name + '.pkl')
        stats_dataframe.to_pickle(stats_dataframe_path)
        print("Stats saved in: " + stats_dataframe_path)
    ######## END SAVING ########

    ######## BEGIN INFERENCE/VISUALIZATION ########
    inference_config = config['inference']
    if inference_config['do_inference']:
        print("Doing inference...")

        # do one of the possible inference
        # stats = clean_stats(stats) # delete connected components that are too small, and too large, they are not bacilli for sure
        inference = Inference(cropped_images, stats, final_image)
        if inference_config['prediction'] == 'SVM':
            print("Using SVM to do inference...")
            red_boxes, green_boxes = inference.svm_prediction()

        elif inference_config['prediction'] == 'CNN':
            print("Using CNN to do inference...")
            red_boxes, green_boxes, coordinates, predictions = inference.network_prediction()
            for i, pred in enumerate(predictions):
                if pred:
                    plt.scatter(coordinates[i,0], coordinates[i,1], color='green')
                else:
                    plt.scatter(coordinates[i,0], coordinates[i,1], color='red')
            answer = input("Do you want to see the scatter plot of the CNN predictions on the geometric projection? (y/n) ")
            if answer == 'y':
                plt.plot(np.arange(0,np.max(coordinates[:,0]),0.1), (1/1.5)*np.arange(0,np.max(coordinates[:,0]),0.1), color='blue')
                plt.plot(np.arange(0,np.max(coordinates[:,0]),0.1), np.arange(0,np.max(coordinates[:,0]),0.1), color='red')
                plt.xlabel('major axis')
                plt.ylabel('minor axis')
                # we want to produce a legend that specifies that green is bacilli, red is non-bacilli, blue is the line MA/ma = 1.5, and red is the line MA/ma = 1
                legend_elements = [Line2D([0], [0], marker='o', color='w', label='CNN bacilli', markerfacecolor='green', markersize=10),
                                   Line2D([0], [0], marker='o', color='w', label='CNN non-bacilli', markerfacecolor='red', markersize=10),
                                   Line2D([0], [0], color='blue', lw=4, label='MA/ma = 1.5'),
                                   Line2D([0], [0], color='red', lw=4, label='MA/ma = 1')]
                plt.legend(handles = legend_elements)
                plt.show()
            

        elif inference_config['prediction'] == 'STATS':
            print("Using geometric properties to do inference...")
            red_boxes, green_boxes, coordinates = inference.ellipse_brute_prediction()
            answer = input("Do you want to see the scatter plot of the geometric projection? (y/n) ")
            if answer == 'y':
                plt.scatter(coordinates[:,0], coordinates[:,1], label='Objects')
                plt.plot(np.arange(0,np.max(coordinates[:,0]),0.1), (1/1.5)*np.arange(0,np.max(coordinates[:,0]),0.1), color='red', label='MA/ma = 1.5')
                plt.xlabel('major axis')
                plt.ylabel('minor axis')
                plt.legend()
                plt.show()

        answer = input("Do you want to visualize the inference results in napari? (y/n)")
        if answer == 'y' and _has_napari:
            viewer = napari.Viewer()
            viewer.add_image(img, name='Inferenced image')
            viewer.add_shapes(red_boxes, shape_type='rectangle', edge_color='red',
                            face_color='transparent', name='Not bacilli')
            viewer.add_shapes(green_boxes, shape_type='rectangle', edge_color='green',
                            face_color='transparent', name='Bacilli')
            napari.run()
    ######## END INFERENCE/VISUALIZATION ########
        
    ######## BEGIN VISUALIZATION ########
    if _has_napari:
        visualization_config = config['visualization']
        show = visualization_config['show']
        if show:
            if preprocess_config['algorithm'] == 'adaptive_gaussian' or 'adaptive_mean':
                images = [img, whole_img_not_cleaned, final_image, image_boxes]
                strings_names = ['original', 'binarized', 'cleaned binarized','original w/ boxes']
            else:
                images = [img, preprocessed_img, whole_img_not_cleaned, final_image, image_boxes]
                strings_names = ['original', 'sharpened', 'binarized', 'cleaned binarized','original w/ boxes']
            visualize_all_list_napari(images, strings_names)
    # currently opening the napari visualizer stops the execution of the code
    ######## END VISUALIZATION ########
