from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load dataset
data = load_diabetes(as_frame=True)
X, y = data.data, data.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models with cross-validation
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5),
    "Lasso Regression": LassoCV(cv=5, max_iter=10000)
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(name)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    if name == "Ridge Regression" or name == "Lasso Regression":
        print("Best Alpha:", model.alpha_)

    print()
