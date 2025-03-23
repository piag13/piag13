import numpy as np

class KNNClassification :
    def __init__ (self, k =5) :
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit (self, X, y) :
        self.X_train = X
        self.y_train = y

    def predict(self , X ) :
        predictions = []
        for x in X :
           # Calculate distances to all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)

            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]

            # Get the majority vote label
            label = np.bincount(k_nearest_labels).argmax()
            predictions.append(label)
        return np.array(predictions)
    
