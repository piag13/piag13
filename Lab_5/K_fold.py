from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

df = pd.read_csv('data/advertising.csv')
X = df[['TV', 'Radio', 'Newspaper']].values
y = df['Sales'].values
# X = X.reshape(-1, 1)

model = LinearRegression()
scores = cross_val_score(model, X, y, cv = 5, scoring = 'neg_mean_squared_error')
rmse = np.sqrt( -scores)
print(f'RMSE: {np.mean(rmse)}')

scores = cross_val_score(model, X, y, cv = 5, scoring = 'neg_mean_absolute_error')
mae = -scores
print(f'MAE: {np.mean(mae)}')

scores = cross_val_score(model, X, y, cv = 5, scoring = 'r2')
r2 = scores
print(f'R2: {np.mean(r2)}')