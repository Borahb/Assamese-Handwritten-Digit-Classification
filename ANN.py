# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:15:36 2021

@author: Bhaskar
"""

#import libraries
import pandas as pd
import numpy as np
from  tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import tensorflow as tf
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import os
from skimage.io import imread
from torch.autograd import Variable

from customDataset import AssameseDigitsDataset


dataset = AssameseDigitsDataset(csv_file = 'Data/Train.csv', root_dir = 'Data/train',
                                transform = transforms.ToTensor()
                                )

train_set , test_set = torch.utils.data.random_split(dataset,[455, 51])

train_loader = DataLoader(dataset = train_set, batch_size = 32, shuffle=True)
test_loader = DataLoader(dataset = test_set, batch_size = 32, shuffle=True)




import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



# fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size,500)
        self.fc2 = nn.Linear(500, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#model = NN(28,10)
#x = torch.randn(32,28)
#print(model(x).shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameteres

input_size = 784
num_class = 10
learning_rate = 0.001
batch_size = 32
num_epochs = 40

# network
model = NN(input_size=input_size, num_classes = num_class).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


#train network

model = model.float()

for epochs in range(num_epochs):
    for batch_idx ,(data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device=device)
        
        #data to correct shape
        #print(data.shape)
        data = data.reshape(data.shape[0],-1)
        #print(data.shape)
        
        #forward
        scores = model(data.float())
        loss = criterion(scores, targets)
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent
        optimizer.step()
        
        
        
        
        

#check accuracy

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()
    


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)




