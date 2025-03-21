import numpy as np
import pandas as pd

df = pd.read_csv('Data/50_Startups.csv')

df.replace(['New York', 'California', 'Florida'], [0, 1, 2], inplace=True)
df = df.infer_objects(copy=False)
X = df[['R&D Spend', 'Administration', 'Marketing Spend','State']].values
y = df['Profit'].values
def my_train_test_split(X, y, test_size = 0.2, random_state = None):
    X = np.array(X)
    y = np.array(y)
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    
    test_size = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test
X_scaled = (X - X.min())/(X.max()- X.min())
print(X_scaled)

X_train, X_test, y_train, y_test = my_train_test_split(X_scaled, y, 0.2, 42)

print (X_train, X_test)
print (y_train, y_test)
