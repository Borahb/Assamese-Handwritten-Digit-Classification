# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 20:46:39 2021

@author: Admin
"""

import PIL
import os
from PIL import Image


f = r'Data/trainann'

os.listdir(f)


for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((28,28))
    img.save(f_img)