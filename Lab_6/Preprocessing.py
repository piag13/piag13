import numpy as np
import pandas as pd
from K_neighbor import KNNRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('Data/50_Startups.csv')
pd.set_option('future.no_silent_downcasting', True)
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

X_train, X_test, y_train, y_test = my_train_test_split(X_scaled, y, 0.2, 42)

k = 5
model = KNNRegression(k = k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual', alpha=0.7)

min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))

plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit Line')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title(f'Actual vs Predicted Profit (KNN Regression, k={k})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

mse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean squared error: {mse:.2f}")
print(f"r2 score: {r2:.2f}")
print(f"Mean absolute error: {mae:.2f}")