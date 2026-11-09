
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("dataset/wisc_bc_data.csv")
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].map({"M": 0, "B": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "MLE": LogisticRegression(C=np.inf, max_iter=5000),
    "MAP-L1": LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=5000),
    "MAP-L2": LogisticRegression(l1_ratio=0, max_iter=5000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    print(name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Log Loss:", log_loss(y_test, prob))
    print("Coefficients:", np.round(model.coef_, 3))
