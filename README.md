
# Tuberculosis Project [[Project Page]](https://github.com/marinadominguez/TBProject)

## Overview
Tuberculosis remains one of the leading causes of death worldwide. The standard diagnostic tool, especially in low- and middle-income countries, is the microscopic examination of stained sputum smears, which is performed manually and is prone to inefficiencies. This project aims to automate the detection and diagnosis of TB using computer vision techniques and convolutional neural networks to analyze whole-slide images of stained sputum smears.

## Features

- **Image Preprocessing**: Preprocess images to remove noise and enhance tuberculosis bacilli visibility. Thresholding Methods and Morphological Operations.
- **Segmentation**: Segment images to identify tuberculosis bacilli, through connected component analysis.
- **Dataset Creation and Annotation**: Create a dataset of segmented 50x50 px images and annotate them.
- **Classification**: Classify images as containing tuberculosis bacilli or not with the help of a CNN, SVM or a Shape based classifier.
- **Visualization**: Visualize images and annotations.
- **Analysis on Whole Slide Images**: Apply the pipeline to whole slide images.
- **Comparative Analysis Across Patients**: Assess bacilli counts across multiple patients to enhance risk stratification and clinical decision-making. Evaluate bacilli distribution patterns within WSIs to refine severity grading and improve diagnostic accuracy.




## Data
The dataset consists of 86 grayscale whole slide images, stored in .CZI format, of stained sputum smears with a resolution of 112534 × 55519 pixels at 20X magnification. The slides were preprocessed, binarized, and annotated for further classification. The severity of the disease was graded by a trained pathologist on an integer scale from 0 (low risk) to 4 (high risk).

Data should be saved under
```
TB_sample/
```

## Installation

Create an empty environment specifying the python version as "python=3.8" and activate it.


### Install the package `tbdetect`

`tbdetect`, including all required packages, can then be pip-installed by running
```
pip install -e .
```
This ensures the scripts are present locally, which enables you to run the provided Python scripts. Additionally, this allows you to modify the baseline solutions due to the `-e` option. Furthermore, this ensures the latest version is installed.


### Import project

To import the project, run python and then enter the following command:
```
from tbdetect import project
```
This might also take some time.

## Run project

To run the project, run
```
project.main()
```

All parameters can be changed using the interactive interface.

## Further Reading
For more information refer to the Projects  report and presentation:
- [Report](resources/final_report.pdf)
- [Presentation](resources/final_presentation.pdf)

