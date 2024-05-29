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
# batch data with tfds
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

# Start the client

def start_client(id, num_rounds, epochs=1, num_clients=2):
    smlp_local = SimpleMLP()
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 4444))
    lr = 0.01
    x = 0.05
    comms_round = 30
    loss='categorical_crossentropy'
    metrics = ['accuracy']
    build_shape = 784

    
    for round_num in range(num_rounds):
        optimizer = SGD(learning_rate=lr, momentum=0.9)
        local_model = smlp_local.build(build_shape, 10)
        local_model.compile(loss=loss, 
                        optimizer=optimizer, 
                        metrics=metrics)
        print(f"\n--- Round {round_num + 1} ---")

        print(f"Receiving global model weights from the server...")
        global_weights = recv_data(client_socket)
        local_model.set_weights(global_weights)
        data_shard = clientdata(id)
        clientbatch = batch_data(data_shard)

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
    num_rounds = 30
    start_client(client_id, num_rounds, num_clients=num_clients)
