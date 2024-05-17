import tensorflow as tf
import tensorflow_federated as tff
import socket
import pickle
import matplotlib.pyplot as plt

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
        input_spec=client_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Create federated learning process
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()

# Network configuration
HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on

def receive_data(conn):
    data = b''
    while True:
        packet = conn.recv(4096)
        if not packet: break
        data += packet
    return pickle.loads(data)

def send_data(conn, data):
    conn.sendall(pickle.dumps(data))

# Track metrics
accuracy_list = []
loss_list = []

# Start the server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print('Server started and listening')
    
    for round_num in range(10):
        client_updates = []
        for _ in range(2):  # Assume 4 clients
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                client_data = receive_data(conn)
                client_updates.append(client_data)
        
        # Aggregate client updates
        state, metrics = iterative_process.next(state, client_updates)
        print(f'Round {round_num+1}, Metrics={metrics}')
        
        accuracy_list.append(metrics['train']['sparse_categorical_accuracy'])
        loss_list.append(metrics['train']['loss'])
        
        # Send updated model to clients
        for _ in range(2):
            conn, addr = s.accept()
            with conn:
                send_data(conn, state)

# Plotting accuracy and loss vs rounds
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), accuracy_list, marker='o')
plt.title('Accuracy vs. Number of Rounds')
plt.xlabel('Number of Rounds')
plt.ylabel('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), loss_list, marker='o')
plt.title('Loss vs. Number of Rounds')
plt.xlabel('Number of Rounds')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
