import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from keras.models import load_model

#import raw data
train_df = pd.read_csv(r'dataset/fashion-mnist_train.csv')
test_df = pd.read_csv(r'dataset/fashion-mnist_test.csv')

#view the data
#train_df.head()

#split data
train_data = np.array(train_df, dtype='float32') #convert it into np array
test_data = np.array(test_df, dtype='float32')

X_train = train_data[:, 1:] / 255 #exclude 0 column
y_train = train_data[:, 0] 

X_test = test_data[:, 1:] / 255
y_test = test_data[:, 0] 

#validation to get stats from keras
X_train, X_validate, y_train, y_validate = train_test_split(
            X_train, 
            y_train, 
            test_size = 0.2, #20% is going to the validation data
            random_state = 12345 #how the data is split
        )

#preview images
#image = X_train[100, :].reshape((28,28)) #reshapes it back to the orignal pixel dim
#plt.imshow(image)
#plt.show()

###Define the model
im_rows = 28
im_cols = 28
batch_size = 512
im_shape = (im_rows, im_cols, 1)

#more formatting
X_train = X_train.reshape(X_train.shape[0], *im_shape)
X_test = X_test.reshape(X_test.shape[0], *im_shape)
X_validate = X_validate.reshape(X_validate.shape[0], *im_shape)

#shows how num pics and their dims
#print('X_train shape: {}'.format(X_train.shape))
#print('X_test shape: {}'.format(X_test.shape))
#print('X_validate shape: {}'.format(X_validate.shape))

cnn_model = Sequential([
            Conv2D(filters = 32, #num feature detectors
                   kernel_size = (3,3), #feature detector size
                   activation = 'relu', 
                   input_shape = im_shape),
            MaxPooling2D(pool_size = (2,2)), #Pooling Feature detector 
            Dropout(0.2),
            
            Flatten(),
            Dense(32, activation = 'relu'), #num feature map
            Dense(10, activation = 'softmax') #num of outputs
        ])
            
###Compile the model
cnn_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer = Adam(lr=0.001), #lr = learning rate
            metrics = ['accuracy']
        )

###Fit the model
cnn_model.fit(
            X_train, y_train, batch_size = batch_size, #data
            epochs = 10,
            verbose = 1, #how much it prints out while calcing
            validation_data = (X_validate, y_validate) #can be *_test data
        )

#save the model
save_path = 'dataset/fashion_model.h5'
cnn_model.save(save_path)

###Make new prediction
selected_img = 8
new_pred = X_test[selected_img, :]

#preview new image
#new_image = new_pred.reshape((28,28)) 
#plt.imshow(new_image)
#plt.show()

new_pred = np.expand_dims(new_pred, axis = 0)
result = cnn_model.predict(new_pred) #says the one with a label = 0

#save the correct label
prob = 0.0
pred_label = 0
for label in range(10):
    if prob < result[0][label]:
        prob = result[0][label]
        pred_label = label

#check if the y_test label is the same
correct_label = y_test[selected_img]

if correct_label == pred_label:
    print('Correct Prediction: {}'.format(pred_label))
else:
    print('Incorrect Prediction: {}'.format(pred_label))
    print('Real Prediction: {}'.format(correct_label))
    
###Load the model
cnn_model = load_model('dataset/fashion_model.h5')









