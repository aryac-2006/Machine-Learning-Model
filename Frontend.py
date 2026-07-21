import streamlit as st
import pandas as pd
import joblib

# ----------- Page settings -------------
st.set_page_config(
    page_title="House Price Predictor",
    layout="centered"
)

# ---------- Load saved files ---------------
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("house_price_scaler.pkl")
encoded_columns = joblib.load("house_price_columns.pkl")

# --------------- Title ----------------
st.title("House Price Prediction")
st.write(
    "Enter the house details below to predict the price..."
)

# ----------- Input fields -----------
# ------------------------------------------------------------------
area = st.number_input(
    "Area",
    min_value=0.0,
    value=1000.0
)
# ___________________________
bedrooms = st.number_input(
    "Bedrooms",
    min_value=0,
    value=3,
    step=1
)

# ___________________________
bathrooms = st.number_input(
    "Bathrooms",
    min_value=0.0,
    value=2.0
)

# ___________________________
floors = st.number_input(
    "Floors",
    min_value=0.0,
    value=1.0
)

year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2015,
    step=1
)

# ___________________________
location = st.selectbox(
    "Location",
    [
        "Urban",
        "Suburban",
        "Rural"
    ]
)

# ___________________________
condition = st.selectbox(
    "Condition",
    [
        "Excellent",
        "Good",
        "Fair",
        "Poor"
    ]
)

# ___________________________
garage = st.selectbox(
    "Garage",
    [
        "Yes",
        "No"
    ]
)
# ----------------------------------------------------------------

# ---------------- Prediction button -------------------
if st.button("Predict Price"):

    # <<<<<<<< Create input data >>>>>>>>>
    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Floors": [floors],
        "Year Built": [year_built],
        "Location": [location],
        "Condition": [condition],
        "Garage": [garage]
    })

    # <<<<<<<<<<<< Apply One-Hot Encoding >>>>>>>>>>>>>
    input_encoded = pd.get_dummies(
        input_data
    )

    # <<<<<<<<<<<< Match training columns >>>>>>>>>>>>
    input_encoded = input_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    # <<<<<<<<<<<<< Apply Scaling >>>>>>>>>>>>
    input_scaled = scaler.transform(
        input_encoded
    )

    # <<<<<<<<<<<< Make prediction >>>>>>>>>>>>>>
    prediction = model.predict(
        input_scaled
    )

    # <<<<<<<<<<<<<< Display prediction >>>>>>>>>>>
    st.success(
        f"Estimated House Price: ₹{prediction[0]:,.2f}"
    )

    #  <<<<<<<<<<<<<<< Display entered details >>>>>>>>>>>>
    st.subheader("House Details")

    details = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Floors": [floors],
        "Year Built": [year_built],
        "Location": [location],
        "Condition": [condition],
        "Garage": [garage]
    })

    # <<<<<<<<<<<<<< Display table >>>>>>>>>>>>
    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )