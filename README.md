# MDCNN
This is a re-implementation of the paper "SAR Automatic Target Recognition Based on Multiview Deep Learning Framework"
The goal of this repo is try to give an example about how to customize the multi-view input data and solve the multi-view image classification problem using deep learning if the network structure will change according to the number of views:

The repo itself does not include any raw data, the readers need to save their images using the heirarchical structure as below:

Level 0: view1, view2, ..., viewM  
Level 1: class1, class2, ..., classC  
Level 2: 1.bmp, 2.bmp, ..., N.bmp  

Namely, there are C subfolders in each folder "view#", and there are N .bmp files in each subfolder "class#"

**DataPreprocessing.py**: The file is used to store all the images saved in the folders show above as well as their labels in a .h5 file. Images from different views of the same target will have the same class label.

**dataset.py**: The file is used to preprocess the images stored in the .h5 file, so that they can be feed into the DataLoader function provided by pytorch. The data augmentation process is also finished in this file.

**MDCNN**: The network structure shown in the paper. 

**TGARSS2018_main**: The main file for training the network and obtain the validation results.

*Pytorch version*: 1.1.0



