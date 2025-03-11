import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000, bias = True):
        self.lr = learning_rate
        self.epochs = epochs
        self.bias = bias
        self.w = None

    def predict(self, X):
        y_pred = np.dot(X, self.w)
        return y_pred

    def loss_function(self, y_true, y_pred):
        n = y_true.shape[0]
        return (1.0 / n) * (((y_true - y_pred) ** 2).sum())

    def gradient(self, X, y) :
        y_pred = self.predict(X)
        error = y - y_pred
        if self.bias:
            X = np.column_stack([np.ones(X.shape[0]), X])
        n = X.shape[0]
        grad = (- 2.0 / n) * (X.T @ error)
        return grad

    def fit(self, X, y):
        if self.bias:
            X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.zeros(X.shape[1])
        
        losses = []
        for epoch in range(self.epochs):
            grad = self.gradient(X, y)
            self.w = self.w - self.lr * grad
            loss = self.loss_function(X, y)
            losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epochs: {epoch}, Loss: {loss}")
