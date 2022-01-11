# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 21:11:44 2021

@author: Bhaskar
"""
#import libraries
import pandas as pd
import numpy as np
from  tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical




df = pd.read_csv("Data/Train.csv")


img_path = 'Data/train/'


X = np.array([img_to_array(load_img(img_path + df['filename'][i],target_size = (28,28,1), grayscale = True))
              for i in tqdm(range(df.shape[0]))
              
              ]).astype('float32')


y = df['label']


#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25,random_state = 42,)



X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)






model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32)

test_loss, test_acc = model.evaluate(X_test, y_test)

print(test_acc)

model.save('doubledigit.h5')


























