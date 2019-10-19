# -*- coding: utf-8 -*-
"""sml_part3_pravin12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BOTWXzi4eV6GZDLxVNY6LhpllChVsf04
"""

import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
 
#### Code structure #####
# 1. Method main has the baseline code. This method is parameterized with kernel size and feature maps.
#      This way i can call the same method with different parameters as asked in the question.
# 2. Method for plotting the graphs.
# 3. Call the above methods with appropriate parameters as per the requirement.
####### End ############
 
#  Returns model after learning from train dataset. And also reports the test loss and accuracy.
#  Parameters: 1. feature_map - An array of number of feature maps for each layer
#              2. kernel_zie - An array of kernel size for all dimention for example [3, 3] for 3x3
#              3. dispStr: Display string when reporting the test loss and accurace.
def main(feature_map, kernel_size, dispStr):
 
 batch_size = 128
 num_classes = 10
 epochs = 12
 
 # input image dimensions
 img_rows, img_cols = 28, 28
 
 # the data, split between train and test sets
 (x_train, y_train), (x_test, y_test) = mnist.load_data()
 
 if K.image_data_format() == 'channels_first':
     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
     input_shape = (1, img_rows, img_cols)
 else:
     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
     input_shape = (img_rows, img_cols, 1)
 
 x_train = x_train.astype('float32')
 x_test = x_test.astype('float32')
 x_train /= 255
 x_test /= 255
 print('x_train shape:', x_train.shape)
 print(x_train.shape[0], 'train samples')
 print(x_test.shape[0], 'test samples')
 
 # convert class vectors to binary class matrices
 y_train = keras.utils.to_categorical(y_train, num_classes)
 y_test = keras.utils.to_categorical(y_test, num_classes)
 
 model = Sequential()
 print ("kernel_sizes =  ", kernel_size)
 print ("feature_map = ", feature_map)
 
 model.add(Conv2D(feature_map[0], kernel_size=(kernel_size[0], kernel_size[1]),
                 activation='relu',
                 input_shape=input_shape))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Conv2D(feature_map[1], (kernel_size[0], kernel_size[1]), activation='relu'))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Flatten())
 model.add(Dense(120, activation='relu'))
 model.add(Dense(84, activation='relu'))
 
 model.add(Dense(num_classes, activation='softmax'))
 
 # https://keras.io/optimizers/
 model.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
               metrics=['accuracy'])
 
 learn = model.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           validation_data=(x_test, y_test))
 score = model.evaluate(x_test, y_test, verbose=0)
 print ("\n", dispStr)
 print ('      Test loss:', score[0])
 print ('      Test accuracy:', score[1])
 print ("###########################################\n")         
        
 return learn
 
# Plots the Model accuracy and loss  of both train and test dataset.
# Parameters: 1. learn - model which is created from train dataset.        
def plot(learn):
 # Plot training & validation accuracy values
 plt.plot(learn.history['acc'])
 plt.plot(learn.history['val_acc'])
 plt.title('Model accuracy')
 plt.ylabel('Accuracy')
 plt.xlabel('Epoch')
 plt.legend(['Train', 'Test'], loc='center right')
 plt.show()
 
 plt.plot(learn.history['loss'])
 plt.plot(learn.history['val_loss'])
 plt.title('Model loss')
 plt.ylabel('Loss')
 plt.xlabel('Epoch')
 plt.legend(['Train', 'Test'], loc='center right')
 plt.show()
 
#main code, calling methods
print("###### BASELINE #######")
learn=main(feature_map = [6,16], kernel_size=[3,3], dispStr="######## BASELINE ERROR AND ACCURACY ######")
plot(learn)
 
print("###### EXPERIMENT WITH kernel size to 5*5 #######")
learn=main(feature_map = [6,16], kernel_size=[5,5],dispStr="######## ERROR AND ACCURACY OF EXPERIMENT WITH kernel size to 5*5 ######")
plot(learn)
 
print("###### EXPERIMENT WITH kernel size to 5*5 AND MODIFIED FEATURE MAPS #######")
learn=main(feature_map = [20,50],kernel_size=[5,5],dispStr="######## ERROR AND ACCURACY OF EXPERIMENT WITH kernel size to 5*5 AND MODIFIED FEATURE MAPS######")
plot(learn)