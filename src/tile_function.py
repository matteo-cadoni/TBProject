"""
This file contains a function that is the main pipeline for the applying the
computations on a dingle tile inside a smear.

The steps that are performed are:
    - Preprocessing
    - Thresholding
    - Postprocessing
    - Cropping (optional)
    - Interactive labelling (optional)
    - Dataset creation (optional)
    - Inference visualization (optional)
    - Visualization (optional)
"""
import pandas as pd
import os
import napari
import matplotlib.pyplot as plt
from src.utils import *
from src.thresholding import Thresholding
from src.postprocessing import Postprocessing
from src.cropping import Cropping
from src.interactivelabelling import InteractiveLabeling
from src.inference_visualization import Inference
from src.visualization import visualize_all_list_napari, add_bounding_boxes
from matplotlib.lines import Line2D

def tile_pipeline(config, img, loader):
    """This function is the main pipeline for the applying the
     computations on a dingle tile inside a smear.

    parameters
    ----------
    config: dict
        dictionary with all the parameters for the pipeline
    img: numpy array
        image of the tile
    loader: class
        class with path to images
    """
    # Preprocess
    preprocess_config = config['preprocessing']
    preprocessed_img = preprocess(preprocess_config, img)

    # Thresholding
    threshold_config = config['thresholding']
    threshold = Thresholding(preprocessed_img, threshold_config)
    thresholded_img = threshold.apply()

    # Postprocessing
    postprocessing_config = config['postprocessing']
    postprocess = Postprocessing(thresholded_img, postprocessing_config)
    whole_img_not_cleaned, final_image, num_bacilli, stats = postprocess.apply()
    stats = clean_stats(stats)

    # bounding boxes
    image_boxes = add_bounding_boxes(img, stats)

    # Defining the configs for the different steps
    labelling_dataset_config = config['labelling_dataset']
    save_config = config['saving']
    inference_config = config['inference']

    # Cropping
    cropped_images = "no images"
    if labelling_dataset_config['create_dataset'] or save_config['save'] or inference_config['prediction'] == "CNN" or inference_config['prediction'] == "STATS":
        cropping_function = Cropping(img, final_image)
        cropped_images = cropping_function.crop_and_pad()

    # Interactive labelling
    # if cropped images is a string
    if type(cropped_images) == str:
        print("No images, cannot label or save dataset or do inference")
    else:
        if labelling_dataset_config['create_dataset']:
            i_l = InteractiveLabeling(cropped_images)
            labels = i_l.run()

            # dataset creation
            dataframe = pd.DataFrame()
            for i in range(0, labels.shape[0]):
                d = {'image': [cropped_images[i]], 'label': [labels[i]], 'stats': [stats[i]]}
                df2 = pd.DataFrame(d)
                dataframe = pd.concat([dataframe, df2], ignore_index=True)

            # save dataset
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

        # Inference visualization
        inference_config = config['inference']

        if inference_config['do_inference']:
            print("Inference...")
            # do one of the possible inference
            inference = Inference(cropped_images, stats, final_image)
            if inference_config['prediction'] == 'SVM':
                red_boxes, green_boxes = inference.svm_prediction()
            elif inference_config['prediction'] == 'CNN':
                red_boxes, green_boxes, coordinates, predictions = inference.network_prediction()
                for i, pred in enumerate(predictions):
                    if pred:
                        plt.scatter(coordinates[i,0], coordinates[i,1], color='green')
                    else:
                        plt.scatter(coordinates[i,0], coordinates[i,1], color='red')
                answer = input("Do you want to see the scatter plot of the CNN predictions on the geometric projection? (y/n)")
                if answer == 'y':
                    plt.plot(np.arange(0,np.max(coordinates[:,0]),0.1), (1/1.5)*np.arange(0,np.max(coordinates[:,0]),0.1), color='blue')
                    plt.plot(np.arange(0,np.max(coordinates[:,0]),0.1), np.arange(0,np.max(coordinates[:,0]),0.1), color='red')
                    plt.xlabel('major axis')
                    plt.ylabel('minor axis')
                    # we want to produce a legend that specifies that green is bacilli, red is non-bacilli, blue is the line MA/ma = 1.5, and red is the line MA/ma = 1
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Bacilli', markerfacecolor='green',
                                              markersize=10),
                                       Line2D([0], [0], marker='o', color='w', label='Not-Bacilli',
                                              markerfacecolor='red', markersize=10),
                                       Line2D([0], [0], color='blue', lw=4, label='MA/ma = 1.5'),
                                       Line2D([0], [0], color='red', lw=4, label='MA/ma = 1')]
                    plt.legend(handles=legend_elements)
                    plt.show()

            elif inference_config['prediction'] == 'STATS':
                red_boxes, green_boxes, coordinates = inference.ellipse_brute_prediction()
                answer = input("Do you want to see the scatter plot of the geometric projection? (y/n) ")
                if answer == 'y':
                    plt.scatter(coordinates[:, 0], coordinates[:, 1], label='Objects')
                    plt.plot(np.arange(0, np.max(coordinates[:, 0]), 0.1),
                             (1 / 1.5) * np.arange(0, np.max(coordinates[:, 0]), 0.1), color='red', label='MA/ma = 1.5')
                    plt.xlabel('major axis')
                    plt.ylabel('minor axis')
                    plt.legend()
                    plt.show()

            answer = input("Do you want to visualize the inference results in napari? (y/n)")
            if answer == 'y':
                viewer = napari.Viewer()
                viewer.add_image(img, name='Inferenced image')
                viewer.add_shapes(red_boxes, shape_type='rectangle', edge_color='red',
                                  face_color='transparent', name='Not bacilli')
                viewer.add_shapes(green_boxes, shape_type='rectangle', edge_color='green',
                                  face_color='transparent', name='Bacilli')
                napari.run()

    # visualization
    visualization_config = config['visualization']
    show = visualization_config['show']
    if show:
        if preprocess_config['algorithm'] == 'rescale':
            images = [img, whole_img_not_cleaned, final_image, image_boxes]
            strings_names = ['original', 'binarized', 'cleaned binarized', 'original w/ boxes']
        else:
            images = [img, preprocessed_img, whole_img_not_cleaned, final_image, image_boxes]
            algorithm = visualization_config['algorithm']
            strings_names = ['original', f'{algorithm}ened', 'binarized', 'cleaned binarized',
                             'original w/ boxes']
        visualize_all_list_napari(images, strings_names)
