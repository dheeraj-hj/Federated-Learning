import pickle
import struct

num_clients = 2

# Send data with length prefix
def send_data(sock, data):
    serialized_data = pickle.dumps(data)
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



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation


class SimpleMLP:
    @staticmethod
    def build(shape, classes) -> Sequential:
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model