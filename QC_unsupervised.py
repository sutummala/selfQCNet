
# Code created by Sudhakar on Feb 2022
# Data extraction for unsupervised QC of T1w and T2w Rigid and Affine Registrations

import os
import tensorflow as tf
from tensorflow.keras import backend as K
#import tensorflow_addons as tfa
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import nibabel as nib
import numpy as np
import random



data_dir = '/media/sudhakar/SUDHAKAR/Work/Data/IXI-Re' # Path to the subjects data directory

voi_size = [30, 30, 30, 60, 45, 15]
#required_voi_size = [224, 224, 3] # suitable for imagenet trained models
d_s_factor = 2 # downsampling factor

Healthy = []

tag = 'hrT1'

scanner_tag = 'HH'
# preparing training and testing data set

for subject in sorted(os.listdir(data_dir)):
    # healthy images
    #print(subject)
    healthy_path = os.path.join(data_dir, subject, 'mni')
    healthy_images = os.listdir(healthy_path)
        
    for healthy_image in healthy_images:
        if tag in healthy_image and scanner_tag in healthy_image and healthy_image.endswith('reoriented.mni.nii'):
            print(f'{healthy_image}')
            input_image = nib.load(os.path.join(healthy_path, healthy_image))
            input_image_data = np.float32(input_image.get_fdata())
            
            x, y, z =  np.shape(input_image_data)
            required_voi = input_image_data[0:x:d_s_factor, 0:y:d_s_factor, 0:z:d_s_factor]
            if True and len(Healthy) <= 1:
                down_sampled = nib.Nifti1Image(required_voi, input_image.affine)
                nib.save(down_sampled, 'healthy_test.nii.gz')  
            #required_slice = np.ndarray.flatten(required_slice)[:np.prod(required_voi_size)].reshape(required_voi_size)
            #required_slice = (required_slice - np.min(required_slice))/(np.max(required_slice) - np.min(required_slice))
            Healthy.append(np.array(required_voi))
            #print(f'healthy {np.shape(Healthy)}')
    

#np.save('/home/sudhakar/T1-mni', Healthy)

