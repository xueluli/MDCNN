# MDCNN
This is a re-implementation of the paper "SAR Automatic Target Recognition Based on Multiview Deep Learning Framework" using Pytorch.

The goal of this repo is try to give an example about how to customize the multi-view input data and solve the multi-view image classification problem using deep learning if the network structure will change according to the number of views:

The repo itself does not include any raw data, the readers need to save their images using the heirarchical structure as below:

Level 0: view1, view2, ..., viewM
Level 1: class1, class2, ..., classC
Level 2: 1.bmp, 2.bmp, ..., N.bmp

Namely, there are C subfolders in each folder "view#", and there are N .bmp files in each subfolder "class#"




