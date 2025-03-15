import numpy as np
import pandas as pd
from LinearRegression import RidgeRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('advertising.csv')
X = df['TV'].values
y = df['Sales'].values

X = X.reshape(-1, 1)
# print(X)

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = 0.01
epochs = 1000
lambda_ = 0.1

model = RidgeRegression(learning_rate = lr, epochs = epochs, lambda_ = lambda_, bias = True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse, rmse, mae, r2 = model.evaluate(X_test, y_test)
print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.7)
plt.plot(X_test, y_pred, color='red', label='Predicted', linewidth = 2)
plt.title('Actual vs Predicted')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.legend()
plt.show()