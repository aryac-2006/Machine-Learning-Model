import pandas as pd
import numpy as np
import joblib
from pandas.core.indexes import category
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ________Load dataset_______
df = pd.read_csv('heart.csv')

# _______Separate features and target_______
X = df.drop("HeartDisease", axis=1)
Y = df["HeartDisease"]

# _________Identify categorical columns_________
categorical_col = X.select_dtypes(['object']).columns

# _____One-Hot encoding______
X = pd.get_dummies(X,categorical_col, drop_first=False)

# Convert Boolean Values to integers
X = X.astype(int)

# ________Store columns names_______
encoded_columns = X.columns.tolist()

print("<< ----------------- Encoded Independent Features ------------------- >>\n")
print(X.head())
print("\n<< ------------------ Dependent Variable -------------------->>\n")
print(Y.head())

# =========== Feature Scaling =========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# -------Convert back to DataFrame--------
X_scaled = pd.DataFrame(X_scaled, columns = encoded_columns)

# --------------------------------------------------------------------------------------

# _____________Train-Test Split____________
X_train, X_test, Y_train, Y_test = train_test_split(
                         X, Y, test_size=0.20,
                         random_state = 42
)
print("\n--------------::: Train-Test Split Completed Successfully! :::---------------\n")
print("X_train shape: ", X_train.shape)
print("\nX_test shape: ", X_test.shape)
print("\nY_train shape: ", Y_train.shape)
print("\nY_test shape: ", Y_test.shape)

# -------------------------------------------------------------------------------

# ____________Create the Logistic Regression model______________
model = LogisticRegression(max_iter=1000)

# ___________________Train the Model__________________
model.fit(X_train, Y_train)
print("\nLogistic Regression Model Trained Successfully ")

# --------------------------------------------------------------------

# _________Make prediction on the test data___________
y_pred = model.predict(X_test)

print("\nActual Values: ")
print(Y_test.iloc[:10].values)
print("\nPredicted Values: ")
print(y_pred[:10])

# --------------------------------------------------------------------------

# _____________Confusion Matrix____________
cm = confusion_matrix(Y_test, y_pred)

print("\n==============::: Confusion Matrix :::===============\n ")
print(cm)

TN, FP, FN, TP = cm.ravel()
print("\n________Confusion Matrix Values_______")
print("\n1) True Negative: ", TN)
print("\n2) False Positives: ", FP)
print("\n3) False Negatives: ", FN)
print("\n4) True Positives: ", TP)

# -----------------------------------------------------------------------------

# _________________Model Evaluation Metrics_______________
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)

print("\n=============::: Evaluation Metrics :::============")
print("\n1. Accuracy: ", accuracy)
print("\n2. Precision: ", precision)
print("\n3. Recall: ", recall)
print("\n4. F1 score: ", f1)

print("\n<<<<<<<<<<<<<<<<<<<< Classification Report >>>>>>>>>>>>>>>>>>>>>>\n")
print(classification_report(Y_test, y_pred))

# --------------------------------------------------------------------------------

# _____Saving Model And Preprocessing Objects_____
joblib.dump(model, 'heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoded_columns, 'columns.pkl')

# ------------------------------------------------------------------------

# __________Load saved files_________
loaded_model = joblib.load("heart_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
loaded_columns = joblib.load("columns.pkl")

# _______Create one sample patient record______
sample_patient = pd.DataFrame({
    "Age": [55],
    "Sex": ["M"],
    "ChestPainType": ["ASY"],
    "RestingBP": [140],
    "Cholesterol": [250],
    "FastingBS": [0],
    "RestingECG": ["Normal"],
    "MaxHR": [150],
    "ExerciseAngina": ["N"],
    "Oldpeak": [1.0],
    "ST_Slope": ["Up"]
})

# __________Encode categorical columns__________
sample_encoded = pd.get_dummies(sample_patient)

# ________Match training columns_______
sample_encoded = sample_encoded.reindex(
    columns=loaded_columns,
    fill_value=0
)

# _____Apply scaling____
sample_scaled = loaded_scaler.transform(sample_encoded)

# _______Make prediction______
sample_prediction = loaded_model.predict(sample_scaled)

print("\n<<<<<<<<<<<<<<<< Sample Patient Prediction >>>>>>>>>>>>>>>>\n")

if sample_prediction[0] == 1:
    print("Heart Disease: Yes")
else:
    print("Heart Disease: No")

