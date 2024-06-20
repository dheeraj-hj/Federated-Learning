import pickle
import struct
#no of clients
num_clients = 2

# Send data with length prefix
def send_data(sock, data):

    serialized_data = pickle.dumps(data) # data is been serialized
    data_length = len(serialized_data)
    sock.sendall(struct.pack('!I', data_length))
    sock.sendall(serialized_data)

# Receive data with length prefix
def recv_data(sock):
    try:
        raw_data = sock.recv(4)
        if len(raw_data) < 4:
            raise ValueError("Incomplete data length received")
        
        data_length = struct.unpack('!I', raw_data)[0]
        data = b""
        while len(data) < data_length:
            packet = sock.recv(data_length - len(data))
            if not packet:
                raise ValueError("Incomplete data received")
            data += packet
        
        return pickle.loads(data)
    except Exception as e:
        print(f"Error receiving data: {e}")
        return None


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


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
    
import tensorflow as tf
import random
# import random_seed
from sklearn.preprocessing import LabelBinarizer

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_train = x_train.reshape((-1, 28 * 28))
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)

data = list(zip(x_train, y_train))
# random.seed(random_seed)
# random.shuffle(data)

# partition_size = len(data) // num_clients

# partitions = []
    
#     # Split the data into partitions
# for i in range(num_clients):
#     start = i * partition_size
#     if i == num_clients - 1:  # The last partition may contain more items if the division is not exact
#         end = len(data)
#     else:
#         end = start + partition_size
#     partitions.append(data[start:end])

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

def clientdata(client_id):

    return partitions[client_id]





