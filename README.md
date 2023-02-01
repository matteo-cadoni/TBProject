
# Tuberculosis Project [[Project Page]](https://github.com/marinadominguez/TBProject)

## Data

Data should be saved under
```
TB_sample/
```

## Run Project

To run the project, run

```
python3 project.py configs/thresholding.yaml
```

All parameters can be changed from the config file available in configs/thresholding.yaml or using the interactive interface.

## To do:  
- [ ] Check out LINUX commands and find out how much storage is available to us
- [x] Add data augmentation for rotational/translational invariance of our model; cut out smaller images to cut off additional bacilli around the center
- [x] Compute ground truth using area and ellipse from hu moments
- [ ] Assess classification accuracy of ground truth, SVM and CNN model for different sizes of training sets and plot graphs for comparison
- [ ] Train model on randomly sampled subset and using the core set approach and compare differences in accuracy
- [x] In the module, separate computational features from optional visualization features
- [x] Specify required and optional packages (i.e., napari) in setup.cfg using parameteres "install_requires" and "extras_require", respectively; if napari is installed, visualization features should be enabled
- [x] Start writing the 4-pages report
- [ ] Define final training set
- [ ] Finilize pipeline, verify we can do everything, try to remove uninformative connected components
- [ ] Train CNN and SVM models on final training set
- [ ] Cross validate CNN and SVM models
- [ ] Merge Module branch 
