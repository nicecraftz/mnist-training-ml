import numpy as np

class Network:
    def __init__(self):
        # He initialization: varianza dei pesi = 2 / numero di input del layer
        self.W1 = np.random.randn(16, 784) * np.sqrt(2 / 784)
        self.b1 = np.zeros((16, 1))

        self.W2 = np.random.randn(16, 16) * np.sqrt(2 / 16)
        self.b2 = np.zeros((16, 1))

        self.W3 = np.random.randn(10, 16) * np.sqrt(2 / 16)
        self.b3 = np.zeros((10, 1))
    
    def train(self, X_batch, Y_batch, learning_rate, epochs):
        for epoch in range(epochs):
            self.forward(X_batch)
            self.backward(X_batch, Y_batch, learning_rate)
            
    def forward(self, X):
        # Hidden Layer 1 + ReLU
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        
        # Hidden Layer 2 + ReLU
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = np.maximum(0, self.Z2)
        
        # Output Layer + Softmax (con axis=0 per i batch)
        self.Z3 = np.dot(self.W3, self.A2) + self.b3
        esponenziali = np.exp(self.Z3 - np.max(self.Z3, axis=0)) # Stabilità numerica
        self.A3 = esponenziali / np.sum(esponenziali, axis=0, keepdims=True)
    
    def backward(self, X, Y, learning_rate):
        dZ3 = self.A3 - Y
        dW3 = np.dot(dZ3, self.A2.T) / X.shape[1] # Media sul batch
        db3 = np.sum(dZ3, axis=1, keepdims=True) / X.shape[1]
        
        errore_propagato = np.dot(self.W3.T, dZ3)
        
        dZ2 = errore_propagato * (self.Z2 > 0)
        dW2 = np.dot(dZ2, self.A1.T) / X.shape[1]
        db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]
        
        errore_propagato_2 = np.dot(self.W2.T, dZ2)
        dZ1 = errore_propagato_2 * (self.Z1 > 0)
        dW1 = np.dot(dZ1, X.T) / X.shape[1]
        db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3