#LAB 8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
features = data.feature_names
target_names = data.target_names

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Plot decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=features, class_names=target_names, rounded=True, proportion=True)
plt.title("Decision Tree - Breast Cancer")
plt.show()

# Predict new sample
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)
print(f"\nPrediction for sample[0]: {target_names[prediction[0]]}")
