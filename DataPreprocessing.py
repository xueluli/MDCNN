from __future__ import print_function
import cv2
import numpy as np
import glob, os
import sys
import time
import h5py
import random
#from scipy import ndimage
from tqdm import tqdm
from tqdm import trange
import pdb

random.seed(1)
np.random.seed(1)
a = np.random.randn(2,3)

Current_Location = os.path.abspath('.')
Patch_Location = os.path.join(Current_Location,'Multiview SRC')
H5_PATH = os.path.join(Current_Location,'H5')
VIEW_NUMBER = [1,2,3,4,5]
TRAIN_NUMBER = [50,20,10,9,8,7,6,5,4,3,2] 
TEST_NUMBER = 50
CLASS_NUMBER = 10
SNR_dB = [10,20]

def add_noise(signal, snr_db):
    ''' 
    signal: np.ndarray
    snr: float

    returns -> np.ndarray
    '''
    snr = 10.0 ** (snr_db / 10.0)
    # Generate the noise as you did
#    noise = acoustics.generator.white(signal.size).reshape(*signal.shape)
    state = np.random.RandomState()
    noise = state.randn(signal.size).reshape(*signal.shape)
    # work out the current SNR
    current_snr = np.mean(signal) / np.std(noise)

    # scale the noise by the snr ratios (smaller noise <=> larger snr)
    noise *= (current_snr / snr)

    # return the new signal with noise
    return signal + noise


VIEW_PATH_LIST = ['view40-50','view80-100','view260-280','view310-320','view355-5']
CLASS_PATH_LIST = ['01_PICKUP', '02_SUV', '03_BTR70', '04_BRDM2', '05_BMP2', '06_T72', '07_ZSU23-4', '08_2S3', '09_MTLB', '10_D20']

SIZE_INPUT_H = 40
SIZE_INPUT_W = 80
extention = 'bmp'

#for i in range(VIEW_NUMBER):
#    INPUT.append(np.empty(shape=(0, SIZE_INPUT_H, SIZE_INPUT_W, 3), dtype=np.float32))

######## test set #########
test_set = []
test_label = []
for i in range(5):
    print("For text, the view number is {}".format(i+1))
    ViewPatch_StorePath = os.path.join(Patch_Location,VIEW_PATH_LIST[i])
    print(ViewPatch_StorePath)
    for idx, category in enumerate(CLASS_PATH_LIST):
        ClassPatch_StorePath = os.path.join(ViewPatch_StorePath,category)
        total_img = glob.glob(ClassPatch_StorePath + '/*.' + extention)
        test_img = random.sample(total_img, TEST_NUMBER)
        test_set.append(test_img)
        test_label.append(str(str(i) + '_' + str(idx)))

#pdb.set_trace()

for view_number in VIEW_NUMBER:
    h5f_path = os.path.join(H5_PATH, 'test_' + 'view'+str(view_number) + '.h5')
    if not os.path.exists(h5f_path):
       h5fw = h5py.File(h5f_path,'w')
       INPUT = {}
       
       for i in range(view_number):
           INPUT[VIEW_PATH_LIST[i]] = np.empty(shape=(0, SIZE_INPUT_H, SIZE_INPUT_W))      
       LABEL = np.empty(shape=(0),dtype=int)
       
       for i in range(view_number):
           for c in range(CLASS_NUMBER):
               CLASS_ImageSet = test_set[i*CLASS_NUMBER+c]
               
#               pdb.set_trace()
               idx_elem = np.expand_dims(c,0)
               idx_elem = idx_elem.astype('int')
#               pdb.set_trace()
               for ii in range(len(CLASS_ImageSet)):
#               print(test_set[i][j])
                   img_path = CLASS_ImageSet[ii]
                   IMG = cv2.imread(img_path,0)
                   IMG_elem = IMG.astype('float32')
                          
#              pdb.set_trace()
                   INPUT[VIEW_PATH_LIST[i]] = np.append(INPUT[VIEW_PATH_LIST[i]], IMG_elem[np.newaxis, ...], axis=0)
                   if i == 0:
                      LABEL = np.append(LABEL, idx_elem, axis=0)
#               pdb.set_trace()
          
       for i in range(view_number): 
           dset_input = h5fw.create_dataset(name=VIEW_PATH_LIST[i], shape=INPUT[VIEW_PATH_LIST[i]].shape, data=INPUT[VIEW_PATH_LIST[i]], dtype=np.float32)
#              pdb.set_trace()
       dset_label = h5fw.create_dataset(name='Label', shape=LABEL.shape, data=LABEL, dtype=np.int)                          


#pdb.set_trace()        
######## training set #########
for view_number in VIEW_NUMBER:        
    print("The view number is {}".format(view_number))
    
    ############ multiple training sizes
    if view_number == 5:
       for train_number in TRAIN_NUMBER: 
           h5f_path = os.path.join(H5_PATH, 'view'+str(view_number) + '_' + 'trainSize'+ str(train_number) + '.h5')

           if not os.path.exists(h5f_path):
              h5fw = h5py.File(h5f_path, 'w')
              INPUT = {}
              
              for i in range(len(VIEW_PATH_LIST)):
                  INPUT[VIEW_PATH_LIST[i]] = np.empty(shape=(0, SIZE_INPUT_H, SIZE_INPUT_W))      
              LABEL = np.empty(shape=(0),dtype=int)
        
              for idx, category in enumerate(CLASS_PATH_LIST):
                  Label = str(idx)
                  idx_elem = np.expand_dims(idx,0)
                  idx_elem = idx_elem.astype('int')
                  for i in range(view_number):
                      Name = 'V_'+str(i)                      
                      ViewPatch_StorePath = os.path.join(Patch_Location,VIEW_PATH_LIST[i])
                      print(ViewPatch_StorePath)
                      print(category,'Label_{}'.format(Label))
                      ClassPatch_StorePath = os.path.join(ViewPatch_StorePath,category)
                      total_img = glob.glob(ClassPatch_StorePath + '/*.' + extention)
                      select_img = [num for num in total_img if num not in test_set]
#                 select_img = random.sample(total_img, 100)
#                 test_img = random.sample(select_img, 50)
                      train_img = random.sample(select_img, train_number)
                      
                     
                      for img_path in train_img:

                          IMG = cv2.imread(img_path,0)
                          IMG_elem = IMG.astype('float32')
                          idx_elem = idx_elem.astype('int')
                          
#                          pdb.set_trace()
                          INPUT[VIEW_PATH_LIST[i]] = np.append(INPUT[VIEW_PATH_LIST[i]], IMG_elem[np.newaxis, ...], axis=0)
                          if i == 0:
                             LABEL = np.append(LABEL, idx_elem, axis=0)
#              dset_input = []
              
              for i in range(len(VIEW_PATH_LIST)): 
                  dset_input = h5fw.create_dataset(name=VIEW_PATH_LIST[i], shape=INPUT[VIEW_PATH_LIST[i]].shape, data=INPUT[VIEW_PATH_LIST[i]], dtype=np.float32)
#              pdb.set_trace()
              dset_label = h5fw.create_dataset(name='Label', shape=LABEL.shape, data=LABEL, dtype=np.int)                          
                      

    ############ multiple training sizes
    if view_number != 5:
       train_number = 5 
       h5f_path = os.path.join(H5_PATH, 'view'+str(view_number) + '_' + 'trainSize'+ str(train_number) + '.h5')

       if not os.path.exists(h5f_path):
          h5fw = h5py.File(h5f_path, 'w')
          INPUT = {}
          
          for i in range(view_number):
              INPUT[VIEW_PATH_LIST[i]] = np.empty(shape=(0, SIZE_INPUT_H, SIZE_INPUT_W))      
          LABEL = np.empty(shape=(0),dtype=int)
    
          for idx, category in enumerate(CLASS_PATH_LIST):
              Label = str(idx)
              idx_elem = np.expand_dims(idx,0)
              idx_elem = idx_elem.astype('int')
              for i in range(view_number):
                  Name = 'V_'+str(i)                      
                  ViewPatch_StorePath = os.path.join(Patch_Location,VIEW_PATH_LIST[i])
                  print(ViewPatch_StorePath)
                  print(category,'Label_{}'.format(Label))
                  ClassPatch_StorePath = os.path.join(ViewPatch_StorePath,category)
                  total_img = glob.glob(ClassPatch_StorePath + '/*.' + extention)
                  select_img = [num for num in total_img if num not in test_set]
#                 select_img = random.sample(total_img, 100)
#                 test_img = random.sample(select_img, 50)
                  train_img = random.sample(select_img, train_number)
                  
                 
                  for img_path in train_img:

                      IMG = cv2.imread(img_path,0)
                      IMG_elem = IMG.astype('float32')
                      
#                          pdb.set_trace()
                      INPUT[VIEW_PATH_LIST[i]] = np.append(INPUT[VIEW_PATH_LIST[i]], IMG_elem[np.newaxis, ...], axis=0)
                      if i == 0:
                         LABEL = np.append(LABEL, idx_elem, axis=0)
#              dset_input = []
#          pdb.set_trace()
          for i in range(view_number): 
              dset_input = h5fw.create_dataset(name=VIEW_PATH_LIST[i], shape=INPUT[VIEW_PATH_LIST[i]].shape, data=INPUT[VIEW_PATH_LIST[i]], dtype=np.float32)
#              pdb.set_trace()
          dset_label = h5fw.create_dataset(name='Label', shape=LABEL.shape, data=LABEL, dtype=np.int)                          
                
                
            
