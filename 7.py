#LAB 7
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Linear Regression on Boston Housing Dataset ---
def simple_linear_regression():
    X, y = fetch_openml(name="boston", version=1, return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Linear Regression on Boston Housing:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression: Boston Housing")
    plt.grid(True)
    plt.show()

# --- Polynomial Regression on Auto MPG Dataset ---
def simple_polynomial_regression():
    data = fetch_openml(name="autoMpg", version=1, as_frame=True)
    df = data.frame.dropna(subset=["horsepower"])
    X = df[["horsepower"]].astype(float)
    y = data.target.loc[df.index].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    print("\nPolynomial Regression on Auto MPG (Degree=3):")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")

    # Sort for smooth curve
    sorted_idx = X_test.values.flatten().argsort()
    X_sorted = X_test.values.flatten()[sorted_idx]
    y_sorted = y_pred[sorted_idx]

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color="blue", alpha=0.6, label="Actual MPG")
    plt.plot(X_sorted, y_sorted, color="red", linewidth=2, label="Polynomial Fit")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.title("Polynomial Regression: Auto MPG")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run both models
simple_linear_regression()
simple_polynomial_regression()
