# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:39:17 2021

@author: Admin
"""

# importing python package
import pandas as pd
  
# read contents of csv file
file = pd.read_csv("ASSAMESE DIGIT DETECTION.csv")
print("\nOriginal file:")
print(file)
  
# adding header
headerList = ['filename', 'label']
  
# converting data frame to csv
file.to_csv("doubeldigit.csv", header=headerList, index=False)
  
# display modified csv file
file2 = pd.read_csv("gfg2.csv")
print('\nModified file:')
print(file2)