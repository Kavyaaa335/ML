#lab6
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(scale=0.2, size=X.shape)

# Add bias term (column of 1s)
X_bias = np.c_[np.ones(X.shape[0]), X]  # Shape: (100, 2)

# Locally Weighted Regression for a single point
def lwr_predict(X_bias, y, x_query, tau):
    weights = np.exp(-((X - x_query) ** 2) / (2 * tau ** 2))
    W = np.diag(weights)
    theta = np.linalg.pinv(X_bias.T @ W @ X_bias) @ X_bias.T @ W @ y
    return np.array([1, x_query]) @ theta

# Predict for all points
def predict_all(X_bias, y, X, tau):
    return np.array([lwr_predict(X_bias, y, x, tau) for x in X])

# Predict using LWR
tau = 0.5
y_pred = predict_all(X_bias, y, X, tau)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="red", s=10, label="Data Points")
plt.plot(X, y_pred, color="green", linewidth=2, label=f"LWR Prediction (tau={tau})")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Locally Weighted Regression (LWR)")
plt.legend()
plt.grid(True)
plt.show()
