import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('Data/50_Startups.csv')
pd.set_option('future.no_silent_downcasting', True)
df.replace(['New York', 'California', 'Florida'], [0, 1, 2], inplace=True)
df = df.infer_objects(copy=False)
X = df[['R&D Spend', 'Administration', 'Marketing Spend','State']].values
y = df['Profit'].values

X_scaled = (X - X.min())/(X.max()- X.min())

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

k = 5
sk_model = KNeighborsRegressor(n_neighbors = k)
sk_model.fit (X_train, y_train)
sk_y_pred = sk_model.predict(X_test)

print(f'RMSE ( Sklearn ): {np.sqrt(np.mean((y_test - sk_y_pred) ** 2))}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, sk_y_pred, color='blue', label='Predicted vs Actual', alpha=0.7)

min_val = min(min(y_test), min(sk_y_pred))
max_val = max(max(y_test), max(sk_y_pred))

plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit Line')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title(f'Actual vs Predicted Profit (KNN Regression, k={k})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
