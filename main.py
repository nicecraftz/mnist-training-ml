import numpy as np

from dataset import load_images, load_labels, one_hot
from network import Network

DATASET_DIR = 'dataset/mnist'
EPOCHS = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.1

def accuracy(network, X, y):
    network.forward(X)
    predizioni = np.argmax(network.A3, axis=0)
    return np.mean(predizioni == y)

def main():
    X_train = load_images(f'{DATASET_DIR}/train-images.idx3-ubyte') # (784, 60000)
    y_train = load_labels(f'{DATASET_DIR}/train-labels.idx1-ubyte') # (60000,)
    Y_train = one_hot(y_train)                                      # (10, 60000)

    X_test = load_images(f'{DATASET_DIR}/t10k-images.idx3-ubyte')   # (784, 10000)
    y_test = load_labels(f'{DATASET_DIR}/t10k-labels.idx1-ubyte')   # (10000,)

    network = Network()
    n = X_train.shape[1]

    for epoch in range(EPOCHS):
        # Mescola gli esempi a ogni epoca
        acc_train = accuracy(network, X_train, y_train)
        acc_test = accuracy(network, X_test, y_test)
        print(f"Epoca {epoch + 1}/{EPOCHS} - train: {acc_train:.2%} - test: {acc_test:.2%}")
        
        permutazione = np.random.permutation(n)
        X_train, Y_train, y_train = X_train[:, permutazione], Y_train[:, permutazione], y_train[permutazione]

        for inizio in range(0, n, BATCH_SIZE):
            X_batch = X_train[:, inizio:inizio + BATCH_SIZE]
            Y_batch = Y_train[:, inizio:inizio + BATCH_SIZE]
            network.train(X_batch, Y_batch, LEARNING_RATE, epochs=1)

if __name__ == '__main__':
    main()
