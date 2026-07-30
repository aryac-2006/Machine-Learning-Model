import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.linear_model import LinearRegression


# ================= Iris Dataset (Classification) =================

# Load Iris dataset
iris = load_iris()

# Features and Target
X_cls = pd.DataFrame(iris.data, columns=iris.feature_names)
y_cls = iris.target

# Display first 5 rows
print("==================<<<< Iris Dataset >>>>================\n")
print(X_cls.head())

# Split data into training and testing sets
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# Scale the data
scaler_cls = StandardScaler()

X_train_cls = scaler_cls.fit_transform(X_train_cls)
X_test_cls = scaler_cls.transform(X_test_cls)

print("\nIndependent Features (X):")
print(iris.feature_names)

print("\nDependent Feature (y):")
print("Target")

# ================= California Housing Dataset (Regression) =================

# Load California Housing dataset
housing = fetch_california_housing()

# Features and Target
X_reg = pd.DataFrame(housing.data, columns=housing.feature_names)
y_reg = housing.target

# Display first 5 rows
print("\n==============<<<< California Housing Dataset >>>>=============\n")
print(X_reg.head())

# Split data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale the data
scaler_reg = StandardScaler()

X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

print("\nIndependent Features (X):")
print(housing.feature_names)

print("\nDependent Feature (y):")
print("House Price")



# ================= Q2 : Classification Algorithms =================

# Dictionary of Classification Models
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Store Results
classification_results = []

print("\n================ Classification Models =================")

# Train and Evaluate Each Model
for model_name, model in classification_models.items():

    # Train
    model.fit(X_train_cls, y_train_cls)

    # Predict
    y_pred = model.predict(X_test_cls)

    # Accuracy
    accuracy = accuracy_score(y_test_cls, y_pred)

    # Save Result
    classification_results.append([model_name, accuracy])

    print(f"\n========== {model_name} ==========")

    print(f"Accuracy : {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_cls, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test_cls, y_pred))

# Comparison Table
comparison_df = pd.DataFrame(
    classification_results,
    columns=["Model", "Accuracy"]
)

print("\n================ Model Comparison =================")
print(comparison_df)

# Best Classification Model
best_cls = comparison_df.loc[comparison_df["Accuracy"].idxmax()]

print("\n===>>> Best Classification Model:\n")
print(best_cls)


# ================= Q3 : Regression Algorithms =================

# Dictionary of Regression Models
regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor()
}

# Store Results
regression_results = []

print("\n================ Regression Models =================")

# Train and Evaluate Each Model
for model_name, model in regression_models.items():

    # Train
    model.fit(X_train_reg, y_train_reg)

    # Predict
    y_pred = model.predict(X_test_reg)

    # R² Score
    r2 = r2_score(y_test_reg, y_pred)

    # Save Result
    regression_results.append([model_name, r2])

    print(f"\n-:-:-:-:-:-: {model_name} :-:-:-:-:-:-")
    print(f"R² Score : {r2:.4f}")

# Comparison Table
comparison_reg_df = pd.DataFrame(
    regression_results,
    columns=["Model", "R² Score"]
)

print("\n--------------|| Model Comparison ||--------------")
print(comparison_reg_df)

# Best Regression Model
best_reg = comparison_reg_df.loc[comparison_reg_df["R² Score"].idxmax()]

print("\n********* Best Regression Model *********\n")
print(best_reg)