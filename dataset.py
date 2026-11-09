import struct
import numpy as np

def load_images(path):
    with open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16)) # Header big-endian
        assert magic == 2051, f"Magic number errato: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    # (784, n): una colonna per immagine, normalizzata in [0, 1]
    return data.reshape(n, rows * cols).T.astype(np.float32) / 255.0

def load_labels(path):
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Magic number errato: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)

def one_hot(labels, num_classes=10):
    # (10, n): colonna j = vettore one-hot dell'etichetta j
    Y = np.zeros((num_classes, labels.size))
    Y[labels, np.arange(labels.size)] = 1
    return Y
