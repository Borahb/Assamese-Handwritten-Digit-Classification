# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 19:05:19 2022

@author: Admin
"""
import pandas as pd
import tensorflow as tf
import numpy as np
import os 
import cv2
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.layers import LSTM, Bidirectional, Conv2D, Dense, Flatten, MaxPooling2D, TimeDistributed, Reshape
from tensorflow.keras.models import Sequential
import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random import shuffle
from sklearn.model_selection import KFold
from tensorflow.keras import optimizers


img_path = 'Data/train/'
df = pd.read_csv("Data/Train.csv")

X = np.array([img_to_array(load_img(img_path + df['filename'][i],target_size = (100,100,3), grayscale = True))
              for i in tqdm(range(df.shape[0]))
              
              ]).astype('float32')


y = df['label']


#Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.1,random_state = 2,stratify = np.array(y))

print(X_train.shape)
print(y_train.shape)

#Normalization

print(X_train[0])

X_train = X_train.reshape(-1,100,100,3)

X_train /= 255
X_test /= 255


#onehot encoding # Label Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




LR = 0.0001
BATCH_SIZE = 32
MOMENTUM = 0.9
IMG_SIZE = 100
EPOCHS = 30

def change_model(model,new_input_shape=(None,299,299,3)):
    
    model._layers[0].batch_input_shape = new_input_shape
    
    new_model = tensorflow.keras.models.model_from_json(model.to_json())
    
    for layer in new_model.layers:
        try:
            layer.set_weight(model.get_layer(name=layer.name).get_weights())
            print("Loaded layer {}".format(layer.name))
        except:
            print("Could not transfer weights for layer {}".format(layer.name))
        
    return new_model

base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

base_model.trainable = False

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dense(10, activation='sigmoid'))

model = add_model
model.compile(loss='categorical_crossentropy', optimizer = optimizers.SGD(lr=LR, momentum = 0.9), metrics = ['accuracy'])



model.summary()

new_model = change_model(model,new_input_shape=(None, 100, 100, 1))

Model_name = 'new-model-fold' + str(fold)
tensorboard = TensorBoard(log_dir='logs/{}'.format(Model_name))


model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose = 1, epochs= 30, batch_size=32)

