# Importing all Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the Laptop Price Dataset
df = pd.read_csv('laptop_price_dataset_v4.csv')

# Display the first 5 rows
print("----------------------------<<<< First 5 rows >>>>--------------------------\n")
print(df.head(5))

# Display the shape of the dataset
print("\n---------< Shape >----------")
print(df.shape)

# Display the columns of the dataset
print("\n-----------------------------<<< Columns >>>------------------------------\n")
print(df.columns)

# Check Dataset Information
print("\n-------------<<<< Information >>>>--------------\n")
print(df.info())

# --------------------------------------------------------------------------

# Find the total number of missing values in each column
print("\n-------< Missing values in each column >------\n")
print(df.isnull().sum())

# Remove rows containing missing values
df = df.dropna()

# Check missing values after removal
print("\n-------< Missing values after removal >------\n")
print(df.isnull().sum())

# Display shape after removing missing values
print("\n>>>>>> Shape after removing missing values <<<<<<\n")
print(df.shape)

# ---------------------------------------------------------------------------

# Identify and remove duplicate rows if any
print("\n--------< Number of Duplicate Rows >--------")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\n-----< Duplicate rows after removal >------")
print(df.duplicated().sum())

# Shape after removing duplicates
print("\n>>>>>> Shape after removing duplicates <<<<<<\n")
print(df.shape)

# -----------------------------------------------------------------------------

# Select Input Features and Target
X = df.drop('price', axis=1)
y = df['price']

# Perform One-Hot Encoding
categorical_col = X.select_dtypes(
    include=['object', 'string', 'category']).columns

X_encoded = pd.get_dummies(X, columns=categorical_col)

# Convert True/False into 1/0
X_encoded = X_encoded.astype(int)

# Store the encoded column names
encoded_columns = X_encoded.columns

# -----------------------------------------------------------------------------

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.33, random_state=42
)

# -----------------------------------------------------------------------------

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------------

# Model Training
model = LinearRegression()
model.fit(
    X_train_scaled,
    y_train
)

# -----------------------------------------------------------------------------

# Make Predictions
y_pred = model.predict(
    X_test_scaled
)

# -----------------------------------------------------------------------------

# Check Model Performance
mse = mean_squared_error(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n===================================================")
print("\nMean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R2 Score: ", r2)
# -----------------------------------------------------------------------------

# Save the Trained Model
joblib.dump(model, 'LR_laptop_price.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(encoded_columns,'columns.pkl')

print("\n-------------::: All files saved successfully :::------------")