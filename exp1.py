
# Q1: Simple Linear Regression using California Housing Dataset

# 1. Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# 2. Load dataset

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

print("First five rows:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
df.info()


# 3. Feature selection

X = df[['AveRooms']]
y = df['Price']


# 4. Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Feature scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert target into NumPy arrays

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# 6. Add intercept column (bias)

X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]


# 7. Linear Regression using Gradient Descent

theta = np.zeros(2)

learning_rate = 0.01
iterations = 3000
m = len(y_train)

for i in range(iterations):

    predictions = X_train_b @ theta

    gradients = (2/m) * X_train_b.T @ (predictions - y_train)

    theta = theta - learning_rate * gradients

y_pred_gd = X_test_b @ theta


# 8. Linear Regression using Normal Equation

theta_ne = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

y_pred_ne = X_test_b @ theta_ne


# 9. Evaluate Gradient Descent

print("\nGradient Descent Results")

print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R2 Score:", r2_score(y_test, y_pred_gd))


# 10. Evaluate Normal Equation

print("\nNormal Equation Results")

print("MSE:", mean_squared_error(y_test, y_pred_ne))
print("R2 Score:", r2_score(y_test, y_pred_ne))


# 11. Visualization

# Sort AveRooms values for plotting smooth lines

X_plot = np.linspace(
    df['AveRooms'].min(),
    df['AveRooms'].max(),
    200
).reshape(-1, 1)

X_plot_scaled = scaler.transform(X_plot)

X_plot_b = np.c_[np.ones((len(X_plot_scaled), 1)), X_plot_scaled]

y_plot_gd = X_plot_b @ theta
y_plot_ne = X_plot_b @ theta_ne

plt.scatter(X_test.flatten(), y_test, color='blue',
            label='Actual Data', alpha=0.4)

plt.plot(X_plot.flatten(), y_plot_gd, color='red',
         label='Gradient Descent')

plt.plot(X_plot.flatten(), y_plot_ne, color='green',
         linestyle='--', label='Normal Equation')

plt.xlabel('Average Rooms')
plt.ylabel('Median House Value')
plt.title('Simple Linear Regression - California Housing')

plt.legend()
plt.show()
