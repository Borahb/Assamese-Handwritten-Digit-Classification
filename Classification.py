# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 21:12:03 2021

@author: Bhaskar
"""

#import libraries
import pandas as pd
import numpy as np
from  tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import tensorflow as tf
import matplotlib.pyplot as plt


df = pd.read_csv("Data/Train.csv")


img_path = 'Data/train/'


X = np.array([img_to_array(load_img(img_path + df['filename'][i],target_size = (28,28,1), grayscale = True))
              for i in tqdm(range(df.shape[0]))
              
              ]).astype('float32')


y = df['label']

print(X.shape,y.shape)

print(X)
#Exploratory Data Analysis

img_index = 0
print(y[img_index])
plt.imshow(X[img_index].reshape(28,28),cmap='Greys')

img_index = 5
print(y[img_index])
plt.imshow(X[img_index].reshape(28,28),cmap='Greys')



#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25,random_state = 42, stratify = np.array(y))


#Normalization

print(X_train[0])

X_train /= 255
X_test /= 255


#Model

input_shape = (28,28,1)
output_class = 10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# define the model
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.3))
model.add(Dense(output_class, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

model.summary()


# train the model
model.fit(x=X_train, y=y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test))


image_index = 5
# print("Original output:",y_test[image_index])
plt.imshow(X_test[image_index].reshape(28,28), cmap='Greys')
pred = model.predict(X_test[image_index].reshape(1,28,28,1))
print("Predicted output:", pred.argmax())













