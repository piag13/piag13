import numpy as np

class LogisticRegression:
    def __init__(self, lr = 0.01, num_iter = 1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def binary_cross_entropy(self, y_true, y_pred):
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    def predict(self, X):
        
    def fit(self, X, y):
        self.X = X
        self.y = y
    