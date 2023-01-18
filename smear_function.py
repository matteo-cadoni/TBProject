from src.preprocess import Preprocessing
from utils import *
from src.thresholding import Thresholding
from src.postprocessing import Postprocessing
from src.visualization import visualize_all_list_napari, add_bounding_boxes, is_blurry_laplacian

    
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
    
    