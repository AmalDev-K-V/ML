import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

# Load Fashion MNIST
data = fetch_openml(
    "Fashion-MNIST",
    version=1,
    as_frame=False,
    parser="liac-arff"
)
X = data.data
y = data.target.astype(int)

# Normalize pixel values
X = X / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, test_size=2000, random_state=42
)

# Test different K values
for k in [3, 5, 7]:
    start = time.time()

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("K =", k)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Time:", round(time.time() - start, 2), "seconds")
