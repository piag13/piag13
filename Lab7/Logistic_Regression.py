import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y_true, y_pred):
        loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))  # Tránh log(0)
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1)) 
        self.bias = 0  
        y = y.reshape((-1, 1)) 
        
        for _ in range(self.epochs):
            linear_output = X @ self.weights + self.bias
            y_pred = self.sigmoid(linear_output)

            dw = (X.T @ (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_output = X @ self.weights + self.bias
        y_pred = self.sigmoid(linear_output)
        return (y_pred > 0.5).astype(int)