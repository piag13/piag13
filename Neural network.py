import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, layer, alpha = 0.1):
        self.layer = layer
        self.alpha = alpha
        self.w = []
        self.b = []
        for i in range(0, len(layer) - 1):
            w_ = np.random.randn(layer[i], layer[i + 1])
            b_ = np.zeros((layer[i + 1], 1))
            self.w.append(w_/layer[i])
            self.b.append(b_)
    def sigmoid(self, X):
        return 1/(1 + np.exp(-X))
    
    def sigmoid_derivative(self, X):
        return X * (1 - X)
    
    def fit(self, X, y):
        A = [X]
        
        out = A[-1]
        for i in range(0, len(self.layer) - 1):
            out = self.sigmoid(out @ self.w[i].T + (self.b.T))
            A.append(out)
        