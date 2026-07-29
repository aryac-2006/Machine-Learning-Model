import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (r2_score, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

# ::::____________<<< Linear Regression >>>____________::::

# <<<<< :::: Load Dataset :::: >>>>>
df = pd.read_csv("House Price Prediction Dataset.csv")

# <<<<< :::: Display First 5 rows :::: >>>>>
print("------------------<<<< First 5 rows >>>> ------------------\n")
print(df.head())

# <<<<< :::: Display the shape :::: >>>>>
print("\n---------<<<< Shape >>>>----------")
print(df.shape)

# <<<<< :::: Display the columns :::: >>>>>
print("\n----------------------<<<< Columns >>>>-----------------------\n")
print(df.columns)

# <<<<< :::: Check Dataset Information :::: >>>>>
print("\n-------------<<<< Information >>>>--------------\n")
print(df.info())

# <<<<< :::: Check Missing Values :::: >>>>>
print("\n-----------<<<< Missing Values >>>>----------\n")
print(df.isnull().sum())

# <<<<< :::: Remove Duplicates :::: >>>>>
df = df.drop_duplicates()

print("\n<<<< Shape After Dropping duplicate values >>>>\n")
print(df.shape)

# <<<<< :::: Remove Id column :::: >>>>>
df = df.drop("Id", axis=1)

# <<<<< :::: One Hot Encoding:::: >>>>>
df = pd.get_dummies(df, drop_first=True)

# <<<<< :::: Select Input Features and Target :::: >>>>>
X = df.drop('Price', axis=1)
y = df['Price']

# <<<<< :::: Train- Test Split :::: >>>>>
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# <<<<< :::: Feature Scaling :::: >>>>>
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# <<<<< :::: Train Linear Regression :::: >>>>>
model = LinearRegression()
model.fit(X_train, y_train)

# <<<<< :::: Make Predictions :::: >>>>>
y_pred = model.predict(X_test)

# <<<<< :::: Actual VS Predicted values :::: >>>>>
result = pd.DataFrame({"Actual Price": y_test,
                       "Predicted Price": y_pred})

print("\n(( Actual Price )) VS (( Predicted Price ))\n")
print(result.head(10))

# <<<<< :::: Evaluate Model :::: >>>>>
r2 = r2_score(y_test, y_pred)
print("\n-->>> R2 Score :", round(r2, 4))


# ::::____________<<< Logistic Regression >>>____________::::

# <<<<< :::: Load Dataset :::: >>>>>
df = pd.read_csv("heart.csv")

# <<<<< :::: Display First 5 rows :::: >>>>>
print("\n------------------<<<< First 5 rows >>>> ------------------\n")
print(df.head())

# <<<<< :::: Display the shape :::: >>>>>
print("\n---------<<<< Shape >>>>----------")
print(df.shape)

# <<<<< :::: Display the columns :::: >>>>>
print("\n----------------------<<<< Columns >>>>-----------------------\n")
print(df.columns)

# <<<<< :::: Check Dataset Information :::: >>>>>
print("\n-------------<<<< Information >>>>--------------\n")
print(df.info())

# <<<<< :::: Check Missing Values :::: >>>>>
print("\n-----------<<<< Missing Values >>>>----------\n")
print(df.isnull().sum())

# <<<<< :::: Remove Duplicates :::: >>>>>
df = df.drop_duplicates()

print("\n<<<< Shape After Dropping duplicate values >>>>\n")
print(df.shape)

# <<<<< :::: One Hot Encoding:::: >>>>>
df = pd.get_dummies(df, drop_first=True)

# <<<<< :::: Select Input Features and Target :::: >>>>>
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# <<<<< :::: Train- Test Split :::: >>>>>
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# <<<<< :::: Feature Scaling :::: >>>>>
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# <<<<< :::: Train Linear Regression :::: >>>>>
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# <<<<< :::: Make Predictions :::: >>>>>
y_pred = model.predict(X_test)

# <<<<< :::: Confusion Matrix :::: >>>>>
cm = confusion_matrix(y_test, y_pred)

print("\n-------------<<<< Confusion Matrix >>>>-----------\n")
print(cm)

# <<<<< :::: Evaluation Metrics :::: >>>>>

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n1) - Accuracy :", round(accuracy, 4))
print("2) - Precision:", round(precision, 4))
print("3) - Recall   :", round(recall, 4))
print("4) - F1 Score :", round(f1, 4))


# ::::________________<<< K-Nearest Neighbors (KNN) >>>__________________::::

from sklearn.neighbors import KNeighborsClassifier

accuracy_list = []
k_values = [3, 5, 7]
print("\n----------<<< Compare accuracies >>>>-----------\n")

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # <<<<< :::: Make Predictions :::: >>>>>
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

    # <<<<< :::: accuracy_score :::: >>>>>
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k = {k} --> Accuracy = {accuracy:.4f}")

best_k = k_values[accuracy_list.index(max(accuracy_list))]
print("\n<> Best K Value :", best_k)
print("<> High Accuracy :", round(max(accuracy_list), 4))


# ::::______________<<< Native Bayes >>>______________::::

from sklearn.naive_bayes import GaussianNB

# <<<<< :::: Train Native Bayes Model :::: >>>>>
model = GaussianNB()
model.fit(X_train, y_train)

# <<<<< :::: Make Predictions :::: >>>>>
y_pred = model.predict(X_test)

# <<<<< :::: Confusion Matrix :::: >>>>>
cm = confusion_matrix(y_test, y_pred)

print("\n-------------<<<< Confusion Matrix >>>>-----------\n")
print(cm)

# <<<<< :::: Classification Report :::: >>>>>
print("\n-------------<<<< Classification Report >>>>-----------\n")
print(classification_report(y_test, y_pred))


# ____________<<< Algorithm Comparison >>>____________

results = []

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

results.append([
    "Logistic Regression",
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
])

# KNN (Best k)
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

results.append([
    f"KNN (k={best_k})",
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
])

# Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

results.append([
    "Naive Bayes",
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
])

# Comparison Table
comparison = pd.DataFrame(
    results,
    columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\n------------<<<< Algorithm Comparison >>>>------------\n")
print(comparison)

# Best Algorithm
best_algorithm = comparison.loc[comparison["Accuracy"].idxmax()]

print("\nBest Algorithm:", best_algorithm["Algorithm"])
print("Highest Accuracy:", round(best_algorithm["Accuracy"], 4))