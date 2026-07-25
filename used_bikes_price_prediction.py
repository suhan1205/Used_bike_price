import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "bike_price_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Used_Bikes.csv")

# -------------------------
# Load Model & Dataset
# -------------------------
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# -------------------------
# Create Encoders
# -------------------------
brand_encoder = LabelEncoder()
city_encoder = LabelEncoder()
owner_encoder = LabelEncoder()

brand_encoder.fit(df["brand"])
city_encoder.fit(df["city"])
owner_encoder.fit(df["owner"])

# -------------------------
# Streamlit Config
# -------------------------
st.set_page_config(
    page_title="Used Bike Price Prediction",
    page_icon="🏍️",
    layout="centered"
)

st.title("🏍️ Used Bike Price Prediction")
st.write("Enter the bike details below.")

# -------------------------
# Inputs
# -------------------------
brand = st.selectbox(
    "Brand",
    sorted(df["brand"].unique())
)

city = st.selectbox(
    "City",
    sorted(df["city"].unique())
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=10000
)

age = st.number_input(
    "Bike Age (Years)",
    min_value=0,
    value=5
)

power = st.number_input(
    "Power (CC)",
    min_value=50,
    value=150
)

owner = st.selectbox(
    "Owner",
    sorted(df["owner"].unique())
)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "city": [city_encoder.transform([city])[0]],
        "kms_driven": [kms_driven],
        "owner": [owner_encoder.transform([owner])[0]],
        "age": [age],
        "power": [power],
        "brand": [brand_encoder.transform([brand])[0]]
    })

    prediction = model.predict(input_df)[0]

    st.balloons()
    st.success(f"💰 Estimated Bike Price: ₹ {prediction:,.0f}")
