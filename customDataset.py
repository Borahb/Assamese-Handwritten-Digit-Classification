# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:27:53 2021

@author: Admin
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
from skimage.transform import resize

class AssameseDigitsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return  len(self.annotations)
    
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path,as_gray=True)
        #image = resize(image,(28,28)).shape
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))


        if  self.transform:
            image = self.transform(image)
            
        return (image, y_label)
    
    
