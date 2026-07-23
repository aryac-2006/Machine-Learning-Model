import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load saved files
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details to predict loan approval.")

# Input Fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Co-applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term (Months)", min_value=12)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Create DataFrame
input_data = pd.DataFrame({
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_term],
    "Credit_History": [credit_history],
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "Property_Area": [property_area]
})

# One-Hot Encoding
input_data = pd.get_dummies(input_data)

# Match training columns
input_data = input_data.reindex(columns=columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")