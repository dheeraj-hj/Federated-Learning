import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import socket
import pickle
import sys

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Get client ID from command-line arguments
client_id = int(sys.argv[1])
num_clients = 2

# Function to create client data in IID fashion
def create_iid_data(client_id, num_clients):
    data_per_client = x_train.shape[0] // num_clients
    start = client_id * data_per_client
    end = start + data_per_client
    client_x = x_train[start:end]
    client_y = y_train[start:end]
    return client_x, client_y

# Function to create client data in Non-IID fashion
def create_non_iid_data(client_id, num_clients, num_shards=10):
    client_data = []
    idx = np.argsort(y_train)
    x_train_sorted = x_train[idx]
    y_train_sorted = y_train[idx]

    shards_per_client = num_shards // num_clients
    shards = np.array_split(np.arange(num_shards), num_clients)
    shard_size = x_train.shape[0] // num_shards

    shard_idxs = shards[client_id]
    for shard_idx in shard_idxs:
        start = shard_idx * shard_size
        end = start + shard_size
        client_data.append((x_train_sorted[start:end], y_train_sorted[start:end]))
    client_x = np.concatenate([d[0] for d in client_data])
    client_y = np.concatenate([d[1] for d in client_data])
    return client_x, client_y

# Choose data distribution method: 'iid' or 'non_iid'
distribution = 'iid'

if distribution == 'iid':
    client_x, client_y = create_iid_data(client_id, num_clients)
else:
    client_x, client_y = create_non_iid_data(client_id, num_clients)

# Convert to TensorFlow dataset
client_data = tf.data.Dataset.from_tensor_slices((client_x, client_y)).batch(10)

# Create a simple model
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=client_data.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Network configuration
HOST = '127.0.0.1'  # Server's hostname or IP address
PORT = 65432        # Server's port

def receive_data(conn):
    data = b''
    while True:
        packet = conn.recv(4096)
        if not packet: break
        data += packet
    return pickle.loads(data)

def send_data(conn, data):
    conn.sendall(pickle.dumps(data))

# Each client participates in federated learning
for round_num in range(10):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        state = receive_data(s)
        
    # Local training
    iterative_process = tff.learning.build_federated_averaging_process(model_fn)
    client_state = iterative_process.next(state, [client_data])
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        send_data(s, client_state)
