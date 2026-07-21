# Streamlit is used to create the web application interface
import streamlit as st

# Pandas used to create and manipulate DataFrames
import pandas as pd

# Joblib is used to load the saved machine learning model
# and preprocessing objects
import joblib

# Load the trained Linear Regression model
model = joblib.load("LR_model.pkl")

# Load the StandardScaler used for feature scaling
scaler = joblib.load("scaler.pkl")

# Load the encoded column names used during model training
encoded_columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Ford Car Price Prediction", layout="centered")
# A suitable page title identifies the application
# The centered layout keeps the interface simple and user-friendly

# Display the title of the application
st.title("Ford Car Price Prediction")
# Display the short description for the user
st.write("Enter the car details below to predict its price")

# Manufacturing year of the car
year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2018,
    step=1
)

# Total mileage of the car
mileage = st.number_input(
    "Mileage",
    min_value=0.0,
    max_value=500000.0,
    value=30000.0,
    step=1000.0
)

# Road tax of the car
tax = st.number_input(
    "Road Tax",
    min_value=0.0,
    max_value=1000.0,
    value=150.0,
    step=10.0
)

# Miles per gallon
mpg = st.number_input(
    "MPG",
    min_value=0.0,
    max_value=200.0,
    value=50.0,
    step=1.0
)

# Engine size in litres
engine_size = st.number_input(
    "Engine Size",
    min_value=0.0,
    max_value=10.0,
    value=1.5,
    step=0.1
)

# ======== Categorical Input Fields =========
# st.selectbox() provides predefined options.
# It prevents invalid user input and makes the application
# easier and more user-friendly.

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual", "Semi-Auto"]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)

# ========= Text Input and Predict Button ==========
# Take the car model name from the user
model_name = st.text_input(
    "Car Model Name",
    placeholder="Enter the car model name"
)

# Create a button to start the price prediction
predict_button = st.button(
    "Predict Price"
)

# =========== Creating Input DataFrame and Encoding ============

if predict_button:

    # Create a DataFrame from the user's inputs
    input_data = pd.DataFrame({
        "model": [model_name],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size]
    })

    # Perform One-Hot Encoding on categorical columns
    input_encoded = pd.get_dummies(
        input_data,
        columns=["model", "transmission", "fuelType"]
    )

    # Align input columns with the columns used during model training
    # Missing columns are filled with 0
    # Extra columns are removed
    input_encoded = input_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )
    # ============ Feature Scaling and Prediction =============
    # Identify the numerical columns
    numerical_columns = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]

    # Apply the loaded StandardScaler
    input_encoded[numerical_columns] = scaler.transform(
        input_encoded[numerical_columns]
    )

    # Make prediction using the loaded model
    prediction = model.predict(input_encoded)

    # Get the predicted price
    predicted_price = prediction[0]

    # Display the predicted price
    st.success(
        f"Predicted Selling Price: £{predicted_price:,.2f}"
    )
    # Display the entered details
    display_data = pd.DataFrame({
        "Car Model": [model_name],
        "Year": [year],
        "Mileage": [mileage],
        "Transmission": [transmission],
        "Fuel Type": [fuel_type],
        "Road Tax": [tax],
        "MPG": [mpg],
        "Engine Size": [engine_size]
    })

    st.subheader("Entered Car Details")
    st.dataframe(display_data)

