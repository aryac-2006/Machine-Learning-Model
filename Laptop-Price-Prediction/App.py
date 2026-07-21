import os
import pandas as pd
import streamlit as st
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "LR_laptop_price.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
encoded_columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))


# <<<<<<<<<<<<<< Title >>>>>>>>>>>>>
st.title("Laptop Price Prediction")
st.write(
    "Enter the complete laptop specifications to predict the estimated price."
)

# <<<<<<<<<<<<<<<<< Input Fields >>>>>>>>>>>>>>>>
# ---------------------------------------------------
brand = st.selectbox(
    "Brand",
    [
        "Dell",
        "HP",
        "Lenovo",
        "Asus",
        "Acer",
        "Apple",
        "MSI",
        "Other"
    ]
)

# -------------------------------
model_name = st.text_input(
    "Laptop Model",
    "Inspiron"
)

# -------------------------------
processor = st.selectbox(
    "Processor",
    [
        "Intel Core i3",
        "Intel Core i5",
        "Intel Core i7",
        "Intel Core i9",
        "AMD Ryzen 3",
        "AMD Ryzen 5",
        "AMD Ryzen 7",
        "AMD Ryzen 9",
        "Other"
    ]
)

# ---------------------------------
processor_speed = st.number_input(
    "Processor Speed (GHz)",
    min_value=1.0,
    max_value=6.0,
    value=2.5,
    step=0.1
)

# -------------------------------
ram = st.number_input(
    "RAM (GB)",
    min_value=2,
    max_value=128,
    value=8,
    step=2
)

# -------------------------------
storage_type = st.selectbox(
    "Storage Type",
    [
        "SSD",
        "HDD",
        "SSD + HDD"
    ]
)

# -------------------------------
storage = st.number_input(
    "Storage Capacity (GB)",
    min_value=128,
    max_value=8192,
    value=512,
    step=128
)

# -------------------------------
display_size = st.number_input(
    "Display Size (Inches)",
    min_value=10.0,
    max_value=20.0,
    value=15.6,
    step=0.1
)

# -------------------------------
resolution = st.selectbox(
    "Screen Resolution",
    [
        "1366x768",
        "1920x1080",
        "2560x1440",
        "3840x2160"
    ]
)

# -------------------------------
graphics = st.selectbox(
    "Graphics Card",
    [
        "Integrated",
        "Intel",
        "NVIDIA",
        "AMD",
        "Other"
    ]
)

# -------------------------------
operating_system = st.selectbox(
    "Operating System",
    [
        "Windows",
        "macOS",
        "Linux",
        "Chrome OS",
        "Other"
    ]
)

# -------------------------------
weight = st.number_input(
    "Weight (kg)",
    min_value=0.5,
    max_value=10.0,
    value=1.8,
    step=0.1
)

# -------------------------------
battery_life = st.number_input(
    "Battery Life (Hours)",
    min_value=1.0,
    max_value=30.0,
    value=6.0,
    step=0.5
)

# -------------------------------
screen_type = st.selectbox(
    "Screen Type",
    [
        "IPS",
        "OLED",
        "LED",
        "LCD",
        "Other"
    ]
)

# -------------------------------
touchscreen = st.selectbox(
    "Touchscreen",
    [
        "Yes",
        "No"
    ]
)
# -------------------------------------------------------

# <<<<<<<<<<<<< Prediction Button >>>>>>>>>>>>>>
if st.button("Predict Laptop Price"):

    # <<-------------- Create Input DataFrame ---------------->>
    input_data = pd.DataFrame({
        "Brand": [brand],
        "Model": [model_name],
        "Processor": [processor],
        "Processor Speed": [processor_speed],
        "RAM": [ram],
        "Storage Type": [storage_type],
        "Storage": [storage],
        "Display Size": [display_size],
        "Resolution": [resolution],
        "Graphics": [graphics],
        "Operating System": [operating_system],
        "Weight": [weight],
        "Battery Life": [battery_life],
        "Screen Type": [screen_type],
        "Touchscreen": [touchscreen]
    })

    # <<---------------- One-Hot Encoding ------------------->>
    input_encoded = pd.get_dummies(
        input_data
    )

    # <<--------------- Match Training Columns --------------->>
    input_encoded = input_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    # <<---------------- Scale Input Data ----------------->>
    input_scaled = scaler.transform(
        input_encoded
    )

    # <<----------------- Predict Price ----------------->>
    prediction = model.predict(
        input_scaled
    )

    # <<---------------- Display Result ---------------->>
    st.success(
        f"Estimated Laptop Price: ₹{prediction[0]:,.2f}"
    )

    # <<---------- Display the entered details ------------>>
    display_data = pd.DataFrame({
        "Brand": [brand],
        "Laptop Model": [model_name],
        "Processor": [processor],
        "Processor Speed": [processor_speed],
        "RAM (GB)": [ram],
        "Storage Type": [storage_type],
        "Storage (GB)": [storage],
        "Display Size": [display_size],
        "Resolution": [resolution],
        "Graphics": [graphics],
        "Operating System": [operating_system],
        "Weight (kg)": [weight],
        "Battery Life": [battery_life],
        "Screen Type": [screen_type],
        "Touchscreen": [touchscreen]
    })

    st.subheader("Entered Laptop Details")
    st.dataframe(
        display_data,
        use_container_width=True
    )