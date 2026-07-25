import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "bike_price_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
DATA_PATH = os.path.join(BASE_DIR, '/Users/sakshamkapoor/Downloads/regression_streamlit/Used_Bikes.csv')

print("BASE_DIR:", BASE_DIR)
print("Files:", os.listdir(BASE_DIR))

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)
df = pd.read_csv(DATA_PATH)

# -------------------------
# Create Label Encoders
# -------------------------
brand_encoder = LabelEncoder()
city_encoder = LabelEncoder()
owner_encoder = LabelEncoder()

brand_encoder.fit(df["brand"])
city_encoder.fit(df["city"])
owner_encoder.fit(df["owner"])

# -------------------------
# Streamlit Page
# -------------------------
st.set_page_config(
    page_title="Used Bike Price Prediction",
    page_icon="🏍️",
    layout="centered"
)

st.title("🏍️ Used Bike Price Prediction")
st.write("Enter bike details to predict its price.")

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

    brand_encoded = brand_encoder.transform([brand])[0]
    city_encoded = city_encoder.transform([city])[0]
    owner_encoded = owner_encoder.transform([owner])[0]

    input_df = pd.DataFrame({
        "city": [city_encoded],
        "kms_driven": [kms_driven],
        "owner": [owner_encoded],
        "age": [age],
        "power": [power],
        "brand": [brand_encoded]
    })

    prediction = model.predict(input_df)[0]
    st.balloons()

    st.success(f"💰 Estimated Bike Price: ₹ {prediction:,.0f}")
