from __future__ import division, print_function, unicode_literals

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

#Resets graph if the kernel was not restarted
tf.reset_default_graph()

#Produce the same outputs
np.random.seed(42)
tf.set_random_seed(42)

#Load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

#***Preview some of the handrawn images
n_samples = 5

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    sample_image = mnist.train.images[index].reshape(28, 28)
    plt.imshow(sample_image, cmap="binary")
    plt.axis("off")

plt.show()

#***The labels of the previewed images
mnist.train.labels[:n_samples]

#~~~~~~~~~INPUT IMAGES

#Create placeholder for the input images
X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")

#~~~~~~~~~PRIMARY CAPSULES

#First layer
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
caps1_n_dims = 8

#Apply 2 convolutional layers to be able to compute the output
conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")

#Squash
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
    
caps1_output = squash(caps1_raw, name="caps1_output")

#~~~~~~~~~DIGIT CAPSULES

#The digit capsule layer contains 10 capsules (one for each digit) of 16 dimensions each
caps2_n_caps = 10
caps2_n_dims = 16

#Create training variable
init_sigma = 0.1

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

#Create first array
batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

#Second array
caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1], name="caps1_output_tiled")

#Shape of first array
W_tiled

#Shape of second array
caps1_output_tiled

#Multiply the 2 arrays and check new shape
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
caps2_predicted

#~~~~~~~~~ROUTING BY AGREEMENT

#Initialize the raw routing weights b_{i,j} to zero
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")

#Round 1
#Calc routing weights with softmax
routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

#Compute the weighted sum of all the predicted output vectors for each second-layer capsule
weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")

#Apply the squash function to get the outputs of the second layer capsules at the end of the first iteration of the routing by agreement algorithm
caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")
caps2_output_round_1

#Round 2
#Look at shapes
caps2_predicted
caps2_output_round_1

#Get the shapes to match
caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
    name="caps2_output_round_1_tiled")

agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

#Update the raw routing weights
raw_weights_round_2 = tf.add(raw_weights, agreement, name="raw_weights_round_2")

#Rest of round2 is same as round 1
routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")

caps2_output = caps2_output_round_2

#~~~~~~~~Static or Dynamic Loop? Would go here



#~~~~~~~~Estimated Class Probabilities (Length)

#Compute norm
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
    
y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")

#Predict class of each instance
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
y_proba_argmax

#Get rid of last 2 dimensions and keep index of longest output vector
y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")
y_pred

#~~~~~~~~~~~~~~~~~~LABELS

#Placeholder for labels
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

#~~~~~~~~~~~~~~~~~MARGIN LOSS

#Use margin looss to detect 2 different digits in one image
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

#Get T for every instanc eof each class [0-9]
T = tf.one_hot(y, depth=caps2_n_caps, name="T")

#Example of this ^^^
with tf.Session():
    print(T.eval(feed_dict={y: np.array([0, 1, 2, 3, 9])}))
    
#Verify shape
caps2_output

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")

#Compute max and reshape
present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                           name="present_error")

# "" ""
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                          name="absent_error")

#Compute loss for each instance and digit
L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")

#Compute sum of digit losses and get mean
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

#~~~~~~~~~~~ RECONSTRUCTION & MASK

#Mask output vectors
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")

#Define reconstruction targets
reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")

#Create reconstr mask and check the shape and compare it 
reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_n_caps,
                                 name="reconstruction_mask")

reconstruction_mask
caps2_output

#Reshape to multiply and apply mask
reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
    name="reconstruction_mask_reshaped")

caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")

caps2_output_masked

#Reshape to flatten decoder inputs
decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_n_caps * caps2_n_dims],
                           name="decoder_input")

decoder_input

#~~~~~~~~~~~~~~~~~~~~~DECODER

#Build decoder
n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 28

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")
    
#~~~~~~~~~~~~~~~~~~~~RECONSTRUCTION LOSS
    
#Compute reconstruction loss
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")

#~~~~~~~~~~~~~~~~~FINAL LOSS

#Calc final loss
alpha = 0.0005

loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

#~~~~~~~~~~~~FINAL TOUCHES

#~~~~~~~~~~~~~ACCURACY

#Measure model's acc
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

#~~~~~~~~~~~~~~~Training Operations

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

#~~~~~~~~~~~~~~~~~~~~Init and Saver

#Add var initializer and saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#~~~~~~~~~~~~~~~~~~~~~~~TRAINING

n_epochs = 10
batch_size = 50
restore_checkpoint = True

n_iterations_per_epoch = mnist.train.num_examples // batch_size
n_iterations_validation = mnist.validation.num_examples // batch_size
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network"

##### comment if already trained
with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch,
                           mask_with_labels: True})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = mnist.validation.next_batch(batch_size)
            loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                               y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val
##### comment if already trained

#~~~~~~~~~~~~~~~~~~~~~EVALUATION
            
n_iterations_test = mnist.test.num_examples // batch_size

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = mnist.test.next_batch(batch_size)
        loss_test, acc_test = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch})
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                  iteration, n_iterations_test,
                  iteration * 100 / n_iterations_test),
              end=" " * 10)
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)
    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
        acc_test * 100, loss_test))

#~~~~~~~~~~~~~~~~~~PREDICTIONS
    
#Fix some images from the test set
n_samples = 5

sample_images = mnist.test.images[:n_samples].reshape([-1, 28, 28, 1])

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    caps2_output_value, decoder_output_value, y_pred_value = sess.run(
            [caps2_output, decoder_output, y_pred],
            feed_dict={X: sample_images,
                       y: np.array([], dtype=np.int64)})
    
#Plot the images and their labels, followed by the corresponding reconstructions and predictions
sample_images = sample_images.reshape(-1, 28, 28)
reconstructions = decoder_output_value.reshape([-1, 28, 28])

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.imshow(sample_images[index], cmap="binary")
    plt.title("Label:" + str(mnist.test.labels[index]))
    plt.axis("off")

plt.show()

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")
    
plt.show()

#Interpreting the Output Vectors

#check shape
caps2_output_value.shape

#tweak pose parameters
def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):
    steps = np.linspace(min, max, n_steps) # -0.25, -0.15, ..., +0.25
    pose_parameters = np.arange(caps2_n_dims) # 0, 1, ..., 15
    tweaks = np.zeros([caps2_n_dims, n_steps, 1, 1, 1, caps2_n_dims, 1])
    tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
    output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
    return tweaks + output_vectors_expanded

n_steps = 11

tweaked_vectors = tweak_pose_parameters(caps2_output_value, n_steps=n_steps)
tweaked_vectors_reshaped = tweaked_vectors.reshape(
    [-1, 1, caps2_n_caps, caps2_n_dims, 1])

#Feed output vectors to the decoders
tweak_labels = np.tile(mnist.test.labels[:n_samples], caps2_n_dims * n_steps)

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    decoder_output_value = sess.run(
            decoder_output,
            feed_dict={caps2_output: tweaked_vectors_reshaped,
                       mask_with_labels: True,
                       y: tweak_labels})
    
#Reshape decoder's output
tweak_reconstructions = decoder_output_value.reshape([caps2_n_dims, n_steps, n_samples, 28, 28])

#Plot all reconstructions
for dim in range(3):
    print("Tweaking output dimension #{}".format(dim))
    plt.figure(figsize=(n_steps / 1.2, n_samples / 1.5))
    for row in range(n_samples):
        for col in range(n_steps):
            plt.subplot(n_samples, n_steps, row * n_steps + col + 1)
            plt.imshow(tweak_reconstructions[dim, col, row], cmap="binary")
            plt.axis("off")
    plt.show()











