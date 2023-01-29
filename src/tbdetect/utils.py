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
from tbdetect.func.visualization import visualize_all_list_napari, add_bounding_boxes, is_blurry_laplacian

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
    if preprocess_config['algorithm'] == "sharp":
        return preprocess.sharpen()
    if preprocess_config['algorithm'] == "rescale":
        return preprocess.rescale()
    
def smear_pipeline(config, smear):
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

                total_number_bacilli += num_bacilli
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

            total_number_bacilli += num_bacilli
            
    print("Total number of bacilli: ", total_number_bacilli)  
    print("Blurry images that were not considered: ", counter_of_blurry_images)
    
    
def tile_pipeline(config, img, loader):
    # Preprocess
    preprocess_config = config['preprocessing']
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
    ######## END POSTPROCESSING ########

    ######## BEGIN ADD BOUNDING BOXES ########
    image_boxes = add_bounding_boxes(img, stats)
    ######## END BOUNDING BOXES ########

    if postprocessing_config['crop']:
        ######## BEGIN CROPPING ########
        cropping_function = Cropping(img, final_image)
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
        ######## END SAVING ########

    ######## BEGIN INFERENCE/VISUALIZATION ########
    inference_config = config['inference']
    if inference_config['do_inference']:
        inference = Inference(cropped_images, stats)
        red_boxes, green_boxes = inference.predict()

        if _has_napari:
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
            if preprocess_config['algorithm'] == 'rescale':
                images = [img, whole_img_not_cleaned, final_image, image_boxes]
                strings_names = ['original', 'binarized', 'cleaned binarized','original w/ boxes']
            else:
                images = [img, preprocessed_img, whole_img_not_cleaned, final_image, image_boxes]
                algorithm = visualization_config['algorithm']
                strings_names = ['original', f'{algorithm}ened', 'binarized', 'cleaned binarized','original w/ boxes']
            visualize_all_list_napari(images, strings_names)
    # currently opening the napari visualizer stops the execution of the code
    ######## END VISUALIZATION ########
