import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, bias=True):
        self.lr = learning_rate
        self.epochs = epochs
        self.bias = bias
        self.w = None

    def predict(self, X):
        if X.shape[1] != self.w.shape[0]:
            X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.w

    def loss_function(self, y_true, y_pred):
        n = y_true.shape[0]
        return (1.0 / n) * (((y_true - y_pred) ** 2).sum())

    def gradient(self, X, y):
        if self.bias and X.shape[1] != len(self.w):
            X = np.c_[np.ones(X.shape[0]), X]
        y_pred = self.predict(X)
        n = X.shape[0]
        grad = (-2.0 / n) * (X.T @ (y - y_pred))
        return grad

    def fit(self, X, y):
        if self.bias:
            X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.zeros(X.shape[1])
        
        losses = []
        for epoch in range(self.epochs):
            grad = self.gradient(X, y)
            self.w = self.w - self.lr * grad
            y_pred = self.predict(X)
            loss = self.loss_function(y, y_pred)
            losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epochs: {epoch}, Loss: {loss}")
    
    def evaluate(self, X, y):
        n = X.shape[0]
        y_pred = self.predict(X)

        mse = np.mean((y - y_pred) ** 2)

        rmse = np.sqrt(np.sum((y - y_pred) ** 2) / n)

        mae = np.mean(np.abs(y - y_pred))

        y_mean = np.mean(y)
        tss = np.sum((y - y_mean) ** 2)

        rss = np.sum((y - y_pred) ** 2)

        r2 = 1 - (rss / tss)
        return mse, rmse, mae, r2


class RidgeRegression(LinearRegression):
    def __init__(self, learning_rate = 0.01, epochs = 1000, lambda_ = 0.1, bias = True):
        super().__init__(learning_rate, epochs, bias)
        self.lambda_ = lambda_

    def loss_function(self, y_true, y_pred):
        n = y_true.shape[0]
        return (1.0 / n) * (((y_true - y_pred) ** 2).sum() + self.lambda_ * (self.w ** 2).sum())

    def gradient(self, X, y):
        y_pred = self.predict(X)
        error = y - y_pred
        grad = (- 2.0 / X.shape[0]) * (X.T @ error) + (2 * self.lambda_ * self.w)
        return grad
    
class LassoRegression(LinearRegression):
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_ = 0.1, bias=True):
        super().__init__(learning_rate, epochs, bias)
        self.lambda_ = lambda_
        
    def loss_function(self, y_true, y_pred):
        n = y_true.shape[0]
        return (1.0 / n) * (((y_true - y_pred) ** 2).sum() + self.lambda_ * ((abs(self.w)).sum()))

    def gradient(self, X, y):
        y_pred = self.predict(X)
        error = y - y_pred
        grad = (- 2.0 / X.shape[0]) * (X.T @ error) + (2 * self.lambda_ * self.w)
        return grad