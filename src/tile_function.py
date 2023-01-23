import pandas as pd
import os
import napari
from src.utils import *
from src.thresholding import Thresholding
from src.postprocessing import Postprocessing
from src.cropping import Cropping
from src.interactivelabelling import InteractiveLabeling
from src.inference_visualization import Inference
from src.visualization import visualize_all_list_napari, add_bounding_boxes


def tile_pipeline(config, img, loader):
    """
    This function is the main pipeline for the tiling process.

    :param config: config file
    :param img: image to be tiled
    :param loader: loader for the image
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

    # bounding boxes
    image_boxes = add_bounding_boxes(img, stats)

    if postprocessing_config['crop']:
        # Cropping
        cropping_function = Cropping(img, final_image)
        cropped_images = cropping_function.crop_and_pad()

    labelling_dataset_config = config['labelling_dataset']
    if labelling_dataset_config['create_dataset']:
        # Interactive labelling
        i_l = InteractiveLabeling(cropped_images)
        labels = i_l.run()

        # dataset creation
        dataframe = pd.DataFrame()
        for i in range(0, labels.shape[0]):
            d = {'image': [cropped_images[i]], 'label': [labels[i]]}

            df2 = pd.DataFrame(d)
            dataframe = pd.concat([dataframe, df2], ignore_index=True)

        # save dataset
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

    # Inference visualization
    inference_config = config['inference']

    if inference_config['do_inference']:
        print("Inference...")
        # do one of the possible inference
        inference = Inference(cropped_images, stats, final_image)
        if inference_config['prediction'] == 'SVM':
            red_boxes, green_boxes = inference.svm_prediction()
        elif inference_config['prediction'] == 'CNN':
            red_boxes, green_boxes = inference.network_prediction()
        elif inference_config['prediction'] == 'STATS':
            red_boxes, green_boxes = inference.stats_prediction()

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
