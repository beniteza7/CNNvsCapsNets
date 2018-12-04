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

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import os

#import raw data
####LOAD ALL PICS####
#UNCOMMENT TO MAKE DATASETS

training_set_size = len(os.listdir('dataset/training_set/dogs'))
test_set_size = len(os.listdir('dataset/test_set/dogs'))

#Load training set
#training_data_file = open('dataset/training_set.csv', 'ab')
for i in range(training_set_size-1): 
    dog_training_image_raw = image.load_img('dataset/training_set/dogs/dog.{}.jpg'.format(i+1), target_size = (64, 64))
    cat_training_image_raw = image.load_img('dataset/training_set/cats/cat.{}.jpg'.format(i+1), target_size = (64, 64))
    
    dog_training_image_raw = image.img_to_array(dog_training_image_raw)
    cat_training_image_raw = image.img_to_array(cat_training_image_raw)
    
    dog_training_image_array = dog_training_image_raw[:,:,0]
    dog_training_image = np.reshape(dog_training_image_array, (1,64*64))
    
    cat_training_image_array = cat_training_image_raw[:,:,0]
    cat_training_image = np.reshape(cat_training_image_array, (1,64*64))
    
    #plt.imshow(dog_training_image_array)
    #plt.show()
   # plt.imshow(cat_training_image_array)
    #plt.show()
    
    labeled_dog_image = np.zeros((1,1 + 64*64))
    labeled_dog_image[:,1:] = dog_training_image
    
    labeled_cat_image = np.ones((1,1 + 64*64))
    labeled_cat_image[:,1:] = cat_training_image
    
#    np.savetxt(training_data_file, labeled_dog_image, fmt="%.0f")
#    np.savetxt(training_data_file, labeled_cat_image, fmt="%.0f")
    
    dog_image_df = pd.DataFrame(labeled_dog_image)
    cat_image_df = pd.DataFrame(labeled_cat_image)
    
    dog_image_df.to_csv('dataset/training_set.csv', mode = 'a', header = False, index = False)
    cat_image_df.to_csv('dataset/training_set.csv', mode = 'a', header = False, index = False)
    
    print('Old training {}'.format(i+1))
    
#training_data_file.close()

#Load test data
#test_data_file = open('dataset/test_set.csv', 'ab')
for i in range(test_set_size-1):
    img_num = i + 1 + 4000
    dog_test_image_raw = image.load_img('dataset/test_set/dogs/dog.{}.jpg'.format(img_num), target_size = (64, 64))
    cat_test_image_raw = image.load_img('dataset/test_set/cats/cat.{}.jpg'.format(img_num), target_size = (64, 64))
    
    dog_test_image_raw = image.img_to_array(dog_test_image_raw)
    cat_test_image_raw = image.img_to_array(cat_test_image_raw)
    
    dog_test_image_array = dog_test_image_raw[:,:,0]
    dog_test_image = np.reshape(dog_test_image_array, (1,64*64))
    
    cat_test_image_array = cat_test_image_raw[:,:,0]
    cat_test_image = np.reshape(cat_test_image_array, (1,64*64))
    
#    plt.imshow(dog_test_image_array)
#    plt.show()
#    plt.imshow(cat_test_image_array)
#    plt.show()
    
    labeled_dog_image = np.zeros((1,1 + 64*64))
    labeled_dog_image[:,1:] = dog_test_image
    
    labeled_cat_image = np.ones((1,1 + 64*64))
    labeled_cat_image[:,1:] = cat_test_image
    
#    np.savetxt(test_data_file, labeled_dog_image, fmt="%.0f")
#    np.savetxt(test_data_file, labeled_cat_image, fmt="%.0f")
    
    dog_image_df = pd.DataFrame(labeled_dog_image)
    cat_image_df = pd.DataFrame(labeled_cat_image)
    
    dog_image_df.to_csv('dataset/test_set.csv', mode = 'a', header = False, index = False)
    cat_image_df.to_csv('dataset/test_set.csv', mode = 'a', header = False, index = False)
    
    print('Old test {}'.format(i+1))

#test_data_file.close()

#########NEW DATA
for i in range(12500): 
    dog_training_image_raw = image.load_img('dataset_2/Dog/{}.jpg'.format(i), target_size = (64, 64))
    cat_training_image_raw = image.load_img('dataset_2/Cat/{}.jpg'.format(i), target_size = (64, 64))
    
    dog_training_image_raw = image.img_to_array(dog_training_image_raw)
    cat_training_image_raw = image.img_to_array(cat_training_image_raw)
    
    dog_training_image_array = dog_training_image_raw[:,:,0]
    dog_training_image = np.reshape(dog_training_image_array, (1,64*64))
    
    cat_training_image_array = cat_training_image_raw[:,:,0]
    cat_training_image = np.reshape(cat_training_image_array, (1,64*64))
    
#    plt.imshow(dog_training_image_array)
#    plt.show()
#    plt.imshow(cat_training_image_array)
#    plt.show()
    
    labeled_dog_image = np.zeros((1,1 + 64*64))
    labeled_dog_image[:,1:] = dog_training_image
    
    labeled_cat_image = np.ones((1,1 + 64*64))
    labeled_cat_image[:,1:] = cat_training_image
    
#    np.savetxt(training_data_file, labeled_dog_image, fmt="%.0f")
#    np.savetxt(training_data_file, labeled_cat_image, fmt="%.0f")
    
    dog_image_df = pd.DataFrame(labeled_dog_image)
    cat_image_df = pd.DataFrame(labeled_cat_image)
    
    if i < 10000:
        dog_image_df.to_csv('dataset/training_set.csv', mode = 'a', header = False, index = False)
        cat_image_df.to_csv('dataset/training_set.csv', mode = 'a', header = False, index = False)
    else:
        dog_image_df.to_csv('dataset/test_set.csv', mode = 'a', header = False, index = False)
        cat_image_df.to_csv('dataset/test_set.csv', mode = 'a', header = False, index = False)
        
    print('New {}'.format(i))


###########


train_df = pd.read_csv(r'dataset/training_set.csv')
test_df = pd.read_csv(r'dataset/test_set.csv')

#view the data
#train_df.head()

#split data
train_data = np.array(train_df, dtype='float32') #convert it into np array
test_data = np.array(test_df, dtype='float32')

#shuffle data
np.random.shuffle(train_data)
np.random.shuffle(test_data)

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
im_rows = 64
im_cols = 64
batch_size = 32     #512
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
            Dense(2, activation = 'softmax') #num of outputs
        ])
            
###Compile the model
cnn_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer = Adam(lr=0.001), #lr = learning rate
            metrics = ['accuracy']
        )

###Fit the model
history = cnn_model.fit(
            X_train, y_train, batch_size = batch_size, #data
            epochs = 30,
            verbose = 1, #how much it prints out while calcing
            validation_data = (X_validate, y_validate) #can be *_test data
        )

#ploting accurace and loss per epoch

#save the model
save_path = 'dataset/doggos_model.h5'
cnn_model.save(save_path)

score = cnn_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plotting the metrics
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')


plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()

fig.savefig('dogs_cnn.pdf')

#####################################Make new prediction
#UNCOMMENT WHEN MODEL DONE

selected_img = 5
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
for label in range(2):
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
cnn_model = load_model('dataset/doggos_model.h5')







