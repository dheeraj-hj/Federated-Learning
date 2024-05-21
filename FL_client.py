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
import socket
import pickle
import struct

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
    
from FL_common import *

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

# Initialize the client model
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def batch_data1(image_list, label_list, bs=32):
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

def create_client_shard(image_list, label_list, num_clients=num_clients, client_id = 1):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = [i for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)  # <- IID
    
    # sort data for non-iid
#     max_y = np.argmax(label_list, axis=-1)
#     sorted_zip = sorted(zip(max_y, label_list, image_list), key=lambda x: x[0])
#     data = [(x,y) for _,y,x in sorted_zip]

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return shards[client_id]

# Start the client
lr = 0.01
comms_round = 50
loss='categorical_crossentropy'
metrics = ['accuracy']
build_shape = 784

def start_client(id, num_rounds, epochs=1, num_clients=2):
    smlp_local = SimpleMLP()
    optimizer = SGD(learning_rate=lr, momentum=0.9)
    local_model = smlp_local.build(build_shape, 10)
    local_model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=metrics)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 4444))

    # Load MNIST data once
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape((-1, 28 * 28))
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    data_shard = create_client_shard(x_train, y_train, num_clients=num_clients , client_id = id)
    # local_data = (x_train[client_id::2], y_train[client_id::2])
    clientbatch = batch_data(data_shard)
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        print(f"Receiving global model weights from the server...")
        global_weights = recv_data(client_socket)
        local_model.set_weights(global_weights)

        # Train the model on local data
        print(f"Training local model on client {id + 1}...")
        local_model.fit(clientbatch, epochs=epochs, verbose=0)
        # scaling_factor = 0.1
        # scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        print(f"Sending updated weights to the server...")
        send_data(client_socket, local_model.get_weights())

    client_socket.close()

if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1])  # Pass client_id as command line argument
    num_rounds = 50
    start_client(client_id, num_rounds, num_clients=num_clients)
