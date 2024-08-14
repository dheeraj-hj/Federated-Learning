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
from pathlib import Path


#no of clients
num_clients = 16
client_frac = 0.25
lr = 0.01 
num_rounds = 20

class CNNModel:
    @staticmethod
    def build(classes):
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


def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss

def select_random_clients(k):
    # Calculate the number of clients to select
    num_clients_to_select = int(num_clients * k)
    # Generate a list of all client indices
    all_clients = list(range(num_clients))
    # Randomly select the desired number of clients
    selected_clients = random.sample(all_clients, num_clients_to_select)
    
    return selected_clients


# partiotioning data among clients

lb = LabelBinarizer()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_train = x_train.reshape((-1, 28, 28, 1))
y_train = lb.fit_transform(y_train)

data = list(zip(x_train, y_train))

random.shuffle(data)

partition_size = len(data) // num_clients

partitions = []
for i in range(num_clients):
    start = i * partition_size
    if i == num_clients - 1:  # The last partition may contain more items if the division is not exact
        end = len(data)
    else:
        end = start + partition_size
    partitions.append(data[start:end])

loss = 'categorical_crossentropy' # categorical_crossentropy because there are 10 classes (-1/n * sum(yi*log(yi_hat))
metrics = ['accuracy'] # Additional metrics to monitor during training and evaluation.
optimizer = SGD(learning_rate=lr, momentum=0.9) # lr -> rate at which it moves to minumum(must be exponentially decreasing), momentum -> parameter to speed up optimization

def client_model_update_generator():
    smlp_local = CNNModel()
    local_model = smlp_local.build(10)
    local_model.compile(loss=loss, 
                        optimizer=optimizer, 
                        metrics=metrics)
    def client_modelupdate(client_id , global_weights):
        local_model.set_weights(global_weights)
        data_shard = partitions[client_id]
        clientbatch = batch_data(data_shard)
        print(f"Training local model on client {client_id + 1}...")
        local_model.fit(clientbatch, epochs=1, verbose=0)
        return local_model.get_weights()
    return client_modelupdate

x_test = x_test.astype('float32') / 255
x_test = x_test.reshape((-1, 28, 28, 1))
y_test = lb.fit_transform(y_test)
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

client_modelupdate = client_model_update_generator()

def main(federated_averaging, file_name='FL_results'):
    smlp_global = CNNModel()
    global_model = smlp_global.build(10)
    global_acc_list = []
    global_loss_list = []
    for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1} ---")
            client_weights_list = []
            global_weights = global_model.get_weights()
            selected_clients_list = select_random_clients(client_frac)
            for client_id in selected_clients_list:
                client_weights = client_modelupdate(client_id, global_weights)
                client_weights_list.append(client_weights)

            print("Performing federated averaging...")
            avgweights = federated_averaging(client_weights_list)
            global_model.set_weights(avgweights)

            # Evaluate global model
            for(X_test, Y_test) in test_batched:
                global_acc, global_loss= test_model(X_test, Y_test, global_model, num_rounds)
                global_acc_list.append(global_acc)
                global_loss_list.append(global_loss)


    print("plotting graph")
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plt.plot(list(range(0, len(global_loss_list))), global_loss_list)
    plt.subplot(122)
    plt.plot(list(range(0, len(global_acc_list))), global_acc_list)
    print('total comm rounds', len(global_acc_list))

    file_name = f"{file_name}_clients{num_clients}_rounds{num_rounds}_frac{client_frac}"
    Path(f'{file_name}.png').parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(f'{file_name}.png')  # Save the plot to a file instead of displaying it

    # write the data to a csv file
    df = pd.DataFrame({'global_acc': global_acc_list, 'global_loss': global_loss_list})
    df.to_csv(f'{file_name}.csv', index=False)




