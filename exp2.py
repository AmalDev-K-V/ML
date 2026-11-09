import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv(r"C:\VSCode\ml\dataset\auto-mpg.csv")

X = df[['displacement']]
y = df['mpg']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

print("Actual MPG:")
print(y_test.head())

print("\nPredicted MPG:")
print(y_pred_linear[:5])

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

results = []

# Linear Regression
y_pred = linear_model.predict(X_test)

results.append([
    "Linear",
    mean_squared_error(y_test, y_pred),
    r2_score(y_test, y_pred)
])

# Polynomial Regression
for degree in [2, 3, 4]:

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    results.append([
        f"Degree {degree}",
        mean_squared_error(y_test, y_pred),
        r2_score(y_test, y_pred)
    ])

results_df = pd.DataFrame(
    results,
    columns=["Model", "MSE", "R2 Score"]
)

print(results_df)    