import streamlit as st
import pandas as pd
import joblib

# _________Load Saved Model_________
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# _______Streamlit Page Configuration_______
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

# _____________Title_____________
st.title("Heart Disease Prediction System")
st.write(
    "Enter the patient's medical information below... "
)

# ___________Input Fields__________
st.header("Patient Information")

# -----------------------------
age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)
# -----------------------------
sex = st.selectbox(
    "Sex",
    ["M", "F"]
)
# -----------------------------
chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)
# -----------------------------
resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value=0,
    max_value=250,
    value=120
)
# -----------------------------
cholesterol = st.number_input(
    "Cholesterol",
    min_value=0,
    max_value=700,
    value=200
)
# -----------------------------
fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)
# -----------------------------
resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)
# -----------------------------
max_hr = st.number_input(
    "Maximum Heart Rate",
    min_value=0,
    max_value=250,
    value=150
)
# -----------------------------
exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["N", "Y"]
)
# -----------------------------
oldpeak = st.number_input(
    "Oldpeak",
    min_value=-5.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)
# -----------------------------
st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)
# -----------------------------------

# ___________________Predict Button_________________
if st.button("Predict", use_container_width=True):

    # -----------Create input DataFrame----------
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    input_encoded = pd.get_dummies(input_data)

    input_encoded = input_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Heart Disease: Yes")
    else:
        st.success("Heart Disease: No")