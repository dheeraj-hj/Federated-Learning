import numpy as np
import pandas as pd
import random
import cv2
import os
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from imutils import paths

# Use 'Agg' backend to avoid Qt dependency issues
matplotlib.use('Agg')  # or 'TkAgg', 'Agg', etc.
debug = 0

def batch_data(image_list, label_list, bs=32):
    '''create a tfds object
    args:
        image_list: a list of numpy arrays of training images
        label_list: a list of binarized labels for each image
        bs: batch size
    return:
        tfds object'''
    #separate shard into data and labels lists
    data = image_list
    label = label_list
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def test_model(X_test, Y_test, model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss

class CNNModel:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        # 32 filters captures basic features , 3x3 kernels convolves and 32 feature maps are generated
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))) 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # 64 filters each 3x3 kernel convolves over 32 feature maps and 64 feature maps are generated
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))#relu -> max(0, x) 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # 128 neurons in the fully connected layer
        model.add(Dense(128, activation='relu'))
        model.add(Dense(classes, activation='softmax')) # softmax -> probability distribution (e^zi / sum(e^zj))
        return model

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images to [0, 1] range
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Reshape the images for CNN input
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Binarize the labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

# Process and batch the training data
train_batched = batch_data(train_images, train_labels)

# Process and batch the test set
test_batched = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(len(test_labels))

lr = 0.01 
comms_round = 50
loss = 'categorical_crossentropy' # categorical_crossentropy because there are 10 classes (-1/n * sum(yi*log(yi_hat))
metrics = ['accuracy'] # Additional metrics to monitor during training and evaluation.
optimizer = SGD(learning_rate=lr, momentum=0.9) # lr -> rate at which it moves to minumum(must be exponentially decreasing), momentum -> parameter to speed up optimization

# Initialize global model
build_shape = (28, 28, 1)
smlp_global = CNNModel()
global_model = smlp_global.build(build_shape, 10)
global_acc_list = []
global_loss_list = []

# Commence global training loop
for comm_round in range(comms_round):
    # Get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    # Train the global model using the entire dataset
    global_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # epochs - number of times the model will iterate over the entire training dataset , verbose 0: No output is displayed during training (silent mode)
    global_model.fit(train_batched, epochs=1, verbose=0) 

    # Test global model and print out metrics after each communication round
    for (X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
        global_acc_list.append(global_acc)
        global_loss_list.append(global_loss)

print("plotting graph")
plt.figure(figsize=(16, 4))
plt.subplot(121)
plt.plot(list(range(0, len(global_loss_list))), global_loss_list)
plt.subplot(122)
plt.plot(list(range(0, len(global_acc_list))), global_acc_list)
print('total comm rounds', len(global_acc_list))

plt.savefig('training_metrics.png')  # Save the plot to a file instead of displaying it
