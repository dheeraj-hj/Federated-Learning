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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from imutils import paths

# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
matplotlib.use('Qt5Agg')  # or 'Qt5Agg', 'Agg', etc.
debug = 0

def load(paths, verbose=-1):
    '''expects images for each class in seperate dir,
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath , cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten() # cv2.imread(imgpath)
        # print(image.shape)
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list
        data.append(image/255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels

    return data, labels


def batch_data(image_list, label_list, bs=32):
    '''create a tfds object
    args:
        image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data = image_list
    label = label_list
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss
    
class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model


# TF_GPU_ALLOCATOR=cuda_malloc_async

# img_path = 'MNIST/trainingSample/trainingSample'
# image_paths = list(paths.list_images(img_path))

# #apply our function
# image_list, label_list = load(image_paths, verbose=10000)

# #binarize the labels
# lb = LabelBinarizer()
# label_list = lb.fit_transform(label_list)

# #split data into training and test set
# X_train, X_test, y_train, y_test = train_test_split(image_list,
#                                                     label_list,
#                                                     test_size=0.1,
#                                                     random_state=42)

# #process and batch the training data
# train_batched = batch_data(X_train , y_train )

# #process and batch the test set
# test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images to [0, 1] range
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Flatten the images
train_images = train_images.reshape((-1, 28 * 28))
test_images = test_images.reshape((-1, 28 * 28))

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
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(learning_rate=lr, momentum=0.9)


#initialize global model

build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST

smlp_global = SimpleMLP()
global_model = smlp_global.build(build_shape, 10)
global_acc_list = []
global_loss_list = []

#commence global training loop
for comm_round in range(comms_round):

    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    # train the global model using the entire dataset
    global_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    global_model.fit(train_batched, epochs=1, verbose=0)

    #test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
        global_acc_list.append(global_acc)
        global_loss_list.append(global_loss)

print("plotting graph")
plt.figure(figsize=(16,4))
plt.subplot(121)
plt.plot(list(range(0,len(global_loss_list))), global_loss_list)
plt.subplot(122)
plt.plot(list(range(0,len(global_acc_list))), global_acc_list)
print('total comm rounds', len(global_acc_list))

plt.show()  # Display the plot
