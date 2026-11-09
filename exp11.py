
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data[:, :2]
y = (data.target == 0).astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear SVM
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Simple visualization
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')

plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            facecolors='none', edgecolors='black',
            s=100, label='Support Vectors')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM")
plt.legend()
plt.show()