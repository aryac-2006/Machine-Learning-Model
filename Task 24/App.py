import streamlit as st
import joblib
import numpy as np

# Load Models
classification_model = joblib.load("best_classification_model.pkl")
regression_model = joblib.load("best_regression_model.pkl")

# Load Scalers
scaler_cls = joblib.load("scaler_classification.pkl")
scaler_reg = joblib.load("scaler_regression.pkl")

st.set_page_config(page_title="Multi Model Prediction", layout="centered")

st.title("Multi Model Prediction App")

problem = st.selectbox(
    "Choose Problem Type",
    ["Classification", "Regression"]
)

if problem == "Classification":

    st.header("Iris Flower Classification")

    sepal_length = st.number_input("Sepal Length", value=5.1)
    sepal_width = st.number_input("Sepal Width", value=3.5)
    petal_length = st.number_input("Petal Length", value=1.4)
    petal_width = st.number_input("Petal Width", value=0.2)

    if st.button("Predict Classification"):

        data = np.array([[sepal_length, sepal_width,
                          petal_length, petal_width]])

        data = scaler_cls.transform(data)

        prediction = classification_model.predict(data)

        classes = ["Setosa", "Versicolor", "Virginica"]

        st.success(f"Predicted Class : {classes[prediction[0]]}")

else:

    st.header("California House Price Prediction")

    MedInc = st.number_input("Median Income", value=3.87)
    HouseAge = st.number_input("House Age", value=28.0)
    AveRooms = st.number_input("Average Rooms", value=5.43)
    AveBedrms = st.number_input("Average Bedrooms", value=1.10)
    Population = st.number_input("Population", value=1425.0)
    AveOccup = st.number_input("Average Occupancy", value=3.07)
    Latitude = st.number_input("Latitude", value=34.05)
    Longitude = st.number_input("Longitude", value=-118.24)

    if st.button("Predict House Price"):

        data = np.array([[MedInc, HouseAge, AveRooms,
                          AveBedrms, Population,
                          AveOccup, Latitude, Longitude]])

        data = scaler_reg.transform(data)
        prediction = regression_model.predict(data)
        house_price = prediction[0] + 100000
        st.success(f"Predicted House Price : {prediction[0]:.2f}")