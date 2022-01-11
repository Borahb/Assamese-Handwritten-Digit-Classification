# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 18:13:44 2022

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

#Tensorflow Version
print("TensorFlow Version:   "+tf.version.VERSION)
print("Keras Version:   "+tf.keras.__version__)

img_path = 'Data/train/'
df = pd.read_csv("Data/Train.csv")

X = np.array([img_to_array(load_img(img_path + df['filename'][i],target_size = (28,28,1), grayscale = True))
              for i in tqdm(range(df.shape[0]))
              
              ]).astype('float32')


y = df['label']

print(X.shape,y.shape)

print(X)

#Exploratory Data Analysis

img_index = 0
print(y_train[img_index])
plt.imshow(X[img_index].reshape(28,28),cmap='Greys')

img_index = 3
print(y[img_index])
plt.imshow(X[img_index].reshape(28,28),cmap='Greys')



#Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.1,random_state = 2,stratify = np.array(y))

print(X_train.shape)
print(y_train.shape)

#Normalization

print(X_train[0])


X_train /= 255
X_test /= 255


#onehot encoding # Label Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Model

model = tf.keras.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                       activation=tf.nn.relu, input_shape = (28,28,1)))
model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                       activation=tf.nn.relu))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))


model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', 
                       activation=tf.nn.relu, input_shape = (28,28,1)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', 
                       activation=tf.nn.relu))
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(256,activation=tf.nn.relu))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(10,activation=tf.nn.softmax))
    
    
    
    
optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    
    
model.compile(optimizer = optimizer, loss='categorical_crossentropy', 
             metrics=["accuracy"])

model.summary()

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                           patience=3,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr=0.00001)

epochs=40
batch_size = 32



datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)







# checking CUDA availability
if(tf.test.is_built_with_cuda() == True):
    print("CUDA Available.. Just wait a few moments...")
else: 
    print("CUDA not Available.. May the force be with you.")
    
#training
    
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_test,y_test),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])    

#saving model
model.save('newmodel.h5')

#testing
image_index = 3
# print("Original output:",y_test[image_index])
plt.imshow(X_test[image_index].reshape(28,28), cmap='Greys')
pred = model.predict(X_test[image_index].reshape(1,28,28,1))
print("Predicted output:", pred.argmax())


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


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
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 





# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_test[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title(" Predicted :{} True :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


#Double digit
#testing
kernel = np.ones((10,10),np.uint8)
kernel1 = np.ones((1,1),np.uint8)


image = cv2.imread('./test11.jpg')
grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey.copy(), 170, 255, cv2.THRESH_BINARY_INV)

thresh = cv2.erode(thresh,kernel1,iterations=1)
thresh = cv2.dilate(thresh,kernel,iterations=2)
plt.imshow(thresh,cmap='gray')
plt.show()
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
preprocessed_digits = []
i = 0
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if ( w <= thresh.shape[0]*(0.05)) and h <= (thresh.shape[1]*(0.05)): # Even after we have eroded and dilated the images, we will still take
                                                                         #precaution not to have small unwanted pixels or features being detected
        
        continue;
    else:
        
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(255, 0, 0), thickness=5)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]


        # Resizing that digit to (18, 18)
        i +=1
        resized_digit = cv2.resize(digit, (18,18))
       
        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)

        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)
print("\n\n\n----------------Contoured Image--------------------")
print('number of digits in the number : ',i)
plt.imshow(image, cmap="gray")
plt.show()
    
inp = np.array(preprocessed_digits)


from tensorflow.keras.models import load_model

model = load_model('newmodel.h5')


final_num = 'The final predicted number is : '
index = 0
for digit in preprocessed_digits:
    prediction = model.predict(digit.reshape(1, 28, 28, 1))   # the first 1 signifies the batch size, the last 1 signifies greyscale image
    plt.imshow(digit.reshape(28, 28), cmap="gray")
    plt.show()
    print("\n\n Prediction of digit number {} : {}".format(index+1,np.argmax(prediction)))
    index +=1
    final_num += str(np.argmax(prediction))
    print ("\n\n---------------------------------------\n\n")
print ("========= FINAL PREDICTED NUMBER ============ \n")
print(final_num)
print ("============================================= \n\n")

model
























