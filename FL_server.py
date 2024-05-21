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
    
# Federated averaging
def federated_averaging(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss


# Start the server
def start_server(num_clients, num_rounds):
    smlp_global = SimpleMLP()
    global_model = smlp_global.build(build_shape, 10)
    global_weights = global_model.get_weights()
    global_acc_list = []
    global_loss_list = []

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # to make the server reuse the port
    server_socket.bind(("localhost",4444))
    server_socket.listen(num_clients)
    print(f"Server is listening for {num_clients} clients...")

    client_sockets = []
    for client_id in range(num_clients):
        client_socket, _ = server_socket.accept()
        client_sockets.append(client_socket)
        print(f"Client {client_id + 1} connected.")

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        client_weights_list = []
        global_weights = global_model.get_weights()
        for client_id, client_socket in enumerate(client_sockets):
            print(f"Sending global model weights to client {client_id + 1}...")
            send_data(client_socket, global_weights)

            print(f"Receiving updated weights from client {client_id + 1}...")
            client_weights = recv_data(client_socket)
            client_weights_list.append(client_weights)

        print("Performing federated averaging...")
        avgweights = federated_averaging(client_weights_list)
        global_model.set_weights(avgweights)

        # Evaluate global model
        for(X_test, Y_test) in test_batched:
            global_acc, global_loss= test_model(X_test, Y_test, global_model, num_rounds)
            global_acc_list.append(global_acc)
            global_loss_list.append(global_loss)

        

    for client_socket in client_sockets:
        client_socket.close()
    server_socket.close()
    return global_acc_list , global_loss_list

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255
x_test = x_test.reshape((-1, 28 * 28))
lb = LabelBinarizer()
y_test = lb.fit_transform(y_test)
build_shape = 784
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

import sys  

if __name__ == "__main__":
    num_rounds = 50
    global_acc_list , global_loss_list = start_server(num_clients, num_rounds)

    
    print("plotting graph")
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    plt.plot(list(range(0,len(global_loss_list))), global_loss_list)
    plt.subplot(122)
    plt.plot(list(range(0,len(global_acc_list))), global_acc_list)
    print('total comm rounds', len(global_acc_list))

    plt.show()  # Display the plot
