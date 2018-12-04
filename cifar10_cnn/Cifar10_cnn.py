#This is a CNN implementation using CNN's
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.constraints import maxnorm #used to constraint the weights incident to each hidden unit to have a norm less than or equal to a desired values
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam # sgd works well with shallow networks, but will try adam model first
from keras import backend as K
#multi TF with multi processing
import tensorflow as tf
import multiprocessing as mp
from keras.utils import np_utils
# loading the cifar10 dataset
from keras.datasets import cifar10

batch_size = 256
num_classes = 10 # 
img_rows = 32
img_cols = 32

#loading the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#data shaping and pre-processing
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255


    
cnn_model = Sequential([
    
	Conv2D(96, (3, 3), activation='relu', padding = 'same', input_shape=(img_rows,img_cols,3)), 
    Dropout(0.2),
    
    Conv2D(96, (3, 3), activation='relu', padding = 'same'),  
    Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2),    
    Dropout(0.5),
    
    Conv2D(192, (3, 3), activation='relu', padding = 'same'),    
    Conv2D(192, (3, 3), activation='relu', padding = 'same'),
    Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2),    
    Dropout(0.5),    
    Conv2D(192, (3, 3), padding = 'same'),
    Activation('relu'),
    Conv2D(192, (1, 1),padding='valid'),
    Activation('relu'),
    Conv2D(10, (1, 1), padding='valid'),
    GlobalAveragePooling2D(),
    Activation('softmax')
])

#train MOdel
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])

cnn_model.summary

#fit model
history = cnn_model.fit(x_train, y_train, batch_size = batch_size, verbose = 1, epochs=300, validation_data=(x_test,y_test),shuffle=True)

score = cnn_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')


plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()

fig.savefig('cifar10_cnn.pdf')