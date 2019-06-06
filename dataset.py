import torch.utils.data as data
import torch
import h5py, cv2
import numpy as np
import random
import torchvision
from PIL import Image
from torchvision import transforms, utils
import pdb

VIEW_PATH_LIST = ['view40-50','view80-100','view260-280','view310-320','view355-5']
class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, view_number=5):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path, 'r')
        if view_number == 1:
            self.input1 = hf.get(VIEW_PATH_LIST[0])
        if view_number == 2:
            self.input1 = hf.get(VIEW_PATH_LIST[0])
            self.input2 = hf.get(VIEW_PATH_LIST[1])
        if view_number == 3:
            self.input1 = hf.get(VIEW_PATH_LIST[0])
            self.input2 = hf.get(VIEW_PATH_LIST[1])
            self.input3 = hf.get(VIEW_PATH_LIST[2])
        if view_number == 4:
            self.input1 = hf.get(VIEW_PATH_LIST[0])
            self.input2 = hf.get(VIEW_PATH_LIST[1])
            self.input3 = hf.get(VIEW_PATH_LIST[2])
            self.input4 = hf.get(VIEW_PATH_LIST[3])
        if view_number == 5:
            self.input1 = hf.get(VIEW_PATH_LIST[0])
            self.input2 = hf.get(VIEW_PATH_LIST[1])
            self.input3 = hf.get(VIEW_PATH_LIST[2])
            self.input4 = hf.get(VIEW_PATH_LIST[3])    
            self.input5 = hf.get(VIEW_PATH_LIST[4])            
        
        self.Nview = view_number
        self.label = hf.get('Label')
#        self.transform = transforms.Normalize([0.485, 0.456], [0.229, 0.224])
        self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([90,90]),
        transforms.ToTensor(),
#        transforms.Normalize([0.485], [0.229])
    ])
#        self.norm = torchvision.transforms.Normalize([0.485, 0.456], [0.229, 0.224])

    def __getitem__(self, index):
#        pdb.set_trace()
        input = []
        if self.Nview == 1: 
           input.append(self.input1[index,:,:]) 
        if self.Nview == 2: 
           input.append(self.input1[index,:,:]) 
           input.append(self.input2[index,:,:])
        if self.Nview == 3: 
           input.append(self.input1[index,:,:]) 
           input.append(self.input2[index,:,:])
           input.append(self.input3[index,:,:])
        if self.Nview == 4: 
           input.append(self.input1[index,:,:])
           input.append(self.input2[index,:,:])
           input.append(self.input3[index,:,:])           
           input.append(self.input4[index,:,:])           
        if self.Nview == 5: 
           input.append(self.input1[index,:,:]) 
           input.append(self.input2[index,:,:])
           input.append(self.input3[index,:,:])           
           input.append(self.input4[index,:,:])
           input.append(self.input5[index,:,:])
           
       
        for i in range(len(input)):
            input[i] = np.expand_dims(input[i],axis=3)
#            input[i] = np.rollaxis(input[i],2)
            input[i] = self.transform(input[i])
            
        Label = self.label[index]
#        Label = np.asarray(Label).reshape(-1)
        # cv2.imwrite('./train_input.png', self.input[index,:,:,:]*255)
        # cv2.imwrite('./train_output.png', self.target[index, :, :, :]*255)
#        pdb.set_trace()s
        return input, Label
        
    def __len__(self):
        return self.input1.shape[0]

