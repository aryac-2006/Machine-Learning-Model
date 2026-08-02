import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AIML Session 23", layout="centered")

#---------------------- Frontend code for Linear Regression -----------------
# Load Linear Regression Files
linear_model = joblib.load("linear_regression_model.pkl")
linear_scaler = joblib.load("linear_scaler.pkl")
linear_columns = joblib.load("linear_columns.pkl")

st.title("Session 23 AIML Models")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["House Price Prediction",
     "Heart Disease - Logistic Regression",
     "Heart Disease - KNN",
     "Heart Disease - Naive Bayes"]
)

if model_choice == "House Price Prediction":

    st.header("House Price Prediction")

    area = st.number_input("Area (sq.ft)", min_value=500, max_value=10000, value=1500)

    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)

    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

    floors = st.number_input("Floors", min_value=1, max_value=5, value=2)

    yearbuilt = st.number_input("Year Built", min_value=1950, max_value=2026, value=2015)

    location = st.selectbox(
        "Location",
        ["Downtown", "Suburban", "Urban", "Rural"]
    )

    condition = st.selectbox(
        "Condition",
        ["Excellent", "Good", "Fair"]
    )

    garage = st.selectbox(
        "Garage",
        ["Yes", "No"]
    )

    if st.button("Predict House Price"):

        input_data = {
            "Area": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floors": floors,
            "YearBuilt": yearbuilt
        }

        input_df = pd.DataFrame([input_data])

        input_df["Location"] = location
        input_df["Condition"] = condition
        input_df["Garage"] = garage

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(columns=linear_columns, fill_value=0)

        input_scaled = linear_scaler.transform(input_df)

        prediction = linear_model.predict(input_scaled)

        st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")

# ---------------------- Frontend code for Logistic Regression -----------------
# Load Heart Disease Files
logistic_model = joblib.load("logistic_regression_model.pkl")
heart_scaler = joblib.load("logistic_scaler.pkl")
heart_columns = joblib.load("heart_columns.pkl")


if model_choice == "Heart Disease - Logistic Regression":

    st.header("Heart Disease Prediction (Logistic Regression)")

    age = st.number_input("Age", 18, 100, 40)

    sex = st.selectbox("Sex", ["M", "F"])

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    resting_bp = st.number_input("Resting Blood Pressure", 80, 250, 120)

    cholesterol = st.number_input("Cholesterol", 0, 700, 200)

    fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "LVH", "ST"]
    )

    max_hr = st.number_input("Maximum Heart Rate", 60, 220, 150)

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"]
    )

    oldpeak = st.number_input(
        "Old Peak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    if st.button("Predict Heart Disease (Logistic)"):

        input_data = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak
        }

        input_df = pd.DataFrame([input_data])

        input_df["Sex"] = sex
        input_df["ChestPainType"] = chest_pain
        input_df["RestingECG"] = resting_ecg
        input_df["ExerciseAngina"] = exercise_angina
        input_df["ST_Slope"] = st_slope

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(columns=heart_columns, fill_value=0)

        input_scaled = heart_scaler.transform(input_df)

        prediction = logistic_model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("Prediction: Heart Disease Detected")
        else:
            st.success("Prediction: No Heart Disease")

# ---------------------- Frontend code for KNN Model -----------------
# Load KNN Model
knn_model = joblib.load("knn_model.pkl")

if model_choice == "Heart Disease - KNN":

    st.header("Heart Disease Prediction (KNN)")

    age = st.number_input("Age", 18, 100, 40, key="k1")

    sex = st.selectbox("Sex", ["M", "F"], key="k2")

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"],
        key="k3"
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        80, 250, 120,
        key="k4"
    )

    cholesterol = st.number_input(
        "Cholesterol",
        0, 700, 200,
        key="k5"
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1],
        key="k6"
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "LVH", "ST"],
        key="k7"
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        60, 220, 150,
        key="k8"
    )

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"],
        key="k9"
    )

    oldpeak = st.number_input(
        "Old Peak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key="k10"
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        key="k11"
    )

    if st.button("Predict Heart Disease (KNN)"):

        input_data = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak
        }

        input_df = pd.DataFrame([input_data])

        input_df["Sex"] = sex
        input_df["ChestPainType"] = chest_pain
        input_df["RestingECG"] = resting_ecg
        input_df["ExerciseAngina"] = exercise_angina
        input_df["ST_Slope"] = st_slope

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(columns=heart_columns, fill_value=0)

        input_scaled = heart_scaler.transform(input_df)

        prediction = knn_model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("Prediction: Heart Disease Detected")
        else:
            st.success("Prediction: No Heart Disease")

# ---------------------- Frontend code for Naive Bayes Model -----------------
# Load Naive Bayes Model
naive_model = joblib.load("naive_bayes_model.pkl")

if model_choice == "Heart Disease - Naive Bayes":

    st.header("Heart Disease Prediction (Naive Bayes)")

    age = st.number_input("Age", 18, 100, 40, key="n1")

    sex = st.selectbox("Sex", ["M", "F"], key="n2")

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"],
        key="n3"
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        80, 250, 120,
        key="n4"
    )

    cholesterol = st.number_input(
        "Cholesterol",
        0, 700, 200,
        key="n5"
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1],
        key="n6"
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "LVH", "ST"],
        key="n7"
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        60, 220, 150,
        key="n8"
    )

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"],
        key="n9"
    )

    oldpeak = st.number_input(
        "Old Peak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key="n10"
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        key="n11"
    )

    if st.button("Predict Heart Disease (Naive Bayes)"):

        input_data = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak
        }

        input_df = pd.DataFrame([input_data])

        input_df["Sex"] = sex
        input_df["ChestPainType"] = chest_pain
        input_df["RestingECG"] = resting_ecg
        input_df["ExerciseAngina"] = exercise_angina
        input_df["ST_Slope"] = st_slope

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(columns=heart_columns, fill_value=0)

        input_scaled = heart_scaler.transform(input_df)

        prediction = naive_model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("Prediction: Heart Disease Detected")
        else:
            st.success("Prediction: No Heart Disease")