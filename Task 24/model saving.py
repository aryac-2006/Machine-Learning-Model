import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression

# ================= Load Datasets =================

from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Classification ----------
iris = load_iris()

X_cls = iris.data
y_cls = iris.target

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

scaler_cls = StandardScaler()

X_train_cls = scaler_cls.fit_transform(X_train_cls)
X_test_cls = scaler_cls.transform(X_test_cls)

# ---------- Regression ----------
housing = fetch_california_housing()

X_reg = housing.data
y_reg = housing.target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()

X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

# ================= Train Best Models =================

best_classification_model = LogisticRegression(max_iter=200)
best_classification_model.fit(X_train_cls, y_train_cls)

best_regression_model = LinearRegression()
best_regression_model.fit(X_train_reg, y_train_reg)

# ================= Save Models =================

joblib.dump(best_classification_model, "best_classification_model.pkl")
joblib.dump(best_regression_model, "best_regression_model.pkl")

joblib.dump(scaler_cls, "scaler_classification.pkl")
joblib.dump(scaler_reg, "scaler_regression.pkl")

joblib.dump(iris.feature_names, "columns_classification.pkl")
joblib.dump(housing.feature_names, "columns_regression.pkl")

print("Best Classification Model Saved Successfully")
print("Best Regression Model Saved Successfully")
print("Scalers Saved Successfully")
print("Columns Saved Successfully")