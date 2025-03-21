import numpy as np

class KNNRegression:
    def __init__(self, k = 5):
        self.k = k
        self.X_train = None
        self.y_train = None
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x)
            k_indices = np.argsort(distances)[:self.k]
            
            k_nearest_values = self.y_train[k_indices]
            mean_value = np.mean(k_nearest_values)
            predictions.append(mean_value)
        
        return np.array(predictions) 
        