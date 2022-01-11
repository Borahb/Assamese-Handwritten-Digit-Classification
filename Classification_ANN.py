# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 21:08:38 2022

@author: Admin
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
import cv2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
sns.set(style="dark",context="notebook",palette="muted")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

img_path = 'Data/train/'
df = pd.read_csv("Data/Train.csv")

X = np.array([img_to_array(load_img(img_path + df['filename'][i],target_size = (28,28,1), grayscale = True))
              for i in tqdm(range(df.shape[0]))
              
              ]).astype('float32')


y = df['label']



#Exploratory Data Analysis

img_index = 0
print(y_train[img_index])
plt.imshow(X_train[img_index].reshape(28,28),cmap='Greys')

img_index = 3
print(y[img_index])
plt.imshow(X[img_index].reshape(28,28),cmap='Greys')



#Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.1,random_state = 42,stratify = np.array(y))

X_train /= 255
X_test /= 255

#onehot encoding # Label Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.reshape(-1,784)
X_test = X_test.reshape(-1,784)

#ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = (784), activation = 'relu', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dropout(0.05))


# Adding the output layer
classifier.add(Dense(units = 10, activation = 'softmax'))


# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, 
                         validation_data = (X_test, y_test), 
                         batch_size = 32, 
                         epochs = 40)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = classifier.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 

classifier.save('annmodel.hf')


#testing
image_index = 3
# print("Original output:",y_test[image_index])
plt.imshow(X_test[image_index].reshape(28,28), cmap='Greys')
pred = classifier.predict(X_test[image_index].reshape(1,784))
print("Predicted output:", pred.argmax())

















