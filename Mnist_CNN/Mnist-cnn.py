import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import backend as K
import os
import tensorflow as tf



# first we are going to load the mnist data set
(x_train,y_train), (x_test,y_test) = mnist.load_data()
#This will be used to splitt the datasets in two parts of the 70,000 images to
#60,000 for the training of the images to 10,000 for testing

#The data set consists of 28x28 images with a batch size of 256 which is
# the number of training examples present in a single batch
img_rows = 28
img_cols =28
batch_size = 256
num_classes = 10
# For reshaping this data we have some if the data where the channel wich can be basically translated
# as the rgb values of image so with channel first the shape of the data is (1,etc,etc)
# channel last (etc , 1 , 1)

if( K.image_data_format() == 'channels_first'):
     x_train = x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
     input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#We must also further reshape the image so that the value of each pixel is an int [0:1] insted of 0;255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('X_train shape:', x_train.shape) #X_train shape: (60000, 28, 28, 1)
print('X_train shape:', x_train.shape) #X_train shape: (60000, 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# We now model our data
cnn_model = Sequential([
    Conv2D(filters=32,  # num feature detectors
           kernel_size=(3, 3),  # feature detector size
           activation='relu',
           input_shape=input_shape),
	Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),  # Pooling Feature detector
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),  # num feature map
	Dropout(0.5),
    Dense(10, activation='softmax')  # num of outputs
])

###Compile the model
cnn_model.compile(
    loss= keras.losses.categorical_crossentropy,
    optimizer=Adam(lr=0.001),  # lr = learning rate
    metrics=['accuracy']
)
###Fit the model
history = cnn_model.fit(
            x_train, y_train, batch_size = batch_size, #data
            epochs = 10,
            verbose = 1, #how much it prints out while calcing
            validation_data = (x_test, y_test)
        )

score = cnn_model.evaluate(x_test, y_test, verbose=0)
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

fig.savefig('mnist_cnn.pdf')