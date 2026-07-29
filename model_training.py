# Importing all Labraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the Ford Car Dataset
df = pd.read_csv('ford_car_dataset.csv')

# Display the first 5 rows
print("----------------------------<<<< Fist 5 rows >>>>--------------------------\n")
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
# Find the total number of missing values in each column.
print("\n-------< Missing values in each column >------\n")
print(df.isnull().sum())
# ---------------------------------------------------------------------------
# Identify and remove duplicate rows if any
print("\n--------< Number of Duplicate Rows >--------")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\n-----< Duplicate rows after removal >------")
print(df.duplicated().sum())
# Shape after removing duplicates
print("\n>>>>>> Shape after removing duplicates <<<<<<\n", df.shape)
# -----------------------------------------------------------------------------

# Select Input Features and Target
X = df.drop('price', axis=1)
y = df['price']

# Perform One-Hot Encoding
categorical_col = X.select_dtypes(include=['object', 'string', 'category']).columns
X_encoded = pd.get_dummies(X, columns = categorical_col)
X_encoded = X_encoded.astype(float)
# Store the encoded column names
encoded_columns = X_encoded.columns

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
                                            X_encoded,
                                                    y,
                                                    test_size = 0.33,
                                                    random_state = 42)

# Numerical columns
numerical_columns = [
    "year",
    "mileage",
    "tax",
    "mpg",
    "engineSize"
]

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test_scaled[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

# Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred = model.predict(X_test_scaled)

# Check Model Performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n===================================================")
print("\nMean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R2 Score: ", r2)

# Save the Trained Model
joblib.dump(model, 'LR_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoded_columns, 'columns.pkl')

print("\n-------------::: All files saved successfully :::------------")
