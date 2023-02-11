
# Tuberculosis Project [[Project Page]](https://github.com/marinadominguez/TBProject)

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


## Data

Data, i.e., the smear images, should be saved under
```
TB_sample/
```



TO DO's:
- [x] Virtual Machine -> mounting the extended storage 

- [x] ELLIPSE: 
	- [x] Is this brute force approach worth it?  Answer this question by plotting: 
	- [x] create a scatter plot for the MA and ma for all of the objects (diagonal are the circular objects)
	- [x] then with the predictions of the neural net paint the dots, and observe if we arrrive to the ideal case of having the two clusters.
	- [x] check whether the nn give us more insights than the plain computer vision ellipse approach
  
- [x] k fold cross validation idea:
  - [x] Plot graphics to assess classification accuracy, with different percentages of training dataset.
  - [x] we need to check what is happening with the loss
  - [x] cross-validation
  - [x] check the different measures
  - [x] plot the distribution certainty of the classifier
  - [x] confussion matrices <--

- [ ] Try the module branch for all the functionalities -- implement all the options
- [ ] polish and finish all the implementations
- [ ] integrate everything together
- [ ] merge branchs
- [x] brief hands-on presentation for the meeting next week
- [ ] write report
- [x] show the plot ( * )
- [ ] we take our clean branch and we implement everything nice and correctly 
- [x] Meeting on Monday to practise 


 - [x] Active learning capabilities 
 	- [x] (expert will explain in the meeting)
	- [x] next tuesday 2pm
	- [ ] play with his tool <- how is it performing in our data, one of us plays the profesional patholoigist role
	- [x] active learning platform
	- [ ] Core set with feature vectors / optimal transport distance.
	- [ ] Assess whether chosen active learning strategy results in better traning
	- [ ] subsample randomly and with active learning strategy -> Train models -> Evaluate differences in accuracy.






### References
https://packaging.python.org/tutorials/packaging-projects/
