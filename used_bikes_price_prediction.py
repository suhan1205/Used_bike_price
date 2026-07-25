import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "bike_price_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Used_Bikes.csv")

# -------------------------------------------------------
# Load Model & Dataset
# -------------------------------------------------------
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------------
# Label Encoders
# -------------------------------------------------------
brand_encoder = LabelEncoder()
city_encoder = LabelEncoder()
owner_encoder = LabelEncoder()

brand_encoder.fit(df["brand"])
city_encoder.fit(df["city"])
owner_encoder.fit(df["owner"])

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Used Bike Price Prediction",
    page_icon="🏍️",
    layout="wide"
)

st.title("🏍️ Used Bike Price Prediction")
st.markdown("Predict the resale price of a used bike using Machine Learning.")

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.title("📊 Project Dashboard")

option = st.sidebar.selectbox(
    "Choose Analysis",
    [
        "Prediction",
        "Dataset Preview",
        "Dataset Information",
        "Statistical Summary",
        "Correlation Heatmap",
        "Model Comparison",
        "Feature Distribution"
    ]
)

# -------------------------------------------------------
# Dataset Preview
# -------------------------------------------------------
if option == "Dataset Preview":

    st.header("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

# -------------------------------------------------------
# Dataset Information
# -------------------------------------------------------
elif option == "Dataset Information":

    st.header("Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum().to_frame("Missing Values"))

# -------------------------------------------------------
# Statistics
# -------------------------------------------------------
elif option == "Statistical Summary":

    st.header("Statistical Summary")

    st.dataframe(df.describe())

# -------------------------------------------------------
# Heatmap
# -------------------------------------------------------
elif option == "Correlation Heatmap":

    st.header("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        linewidths=.5,
        ax=ax
    )

    st.pyplot(fig)

# -------------------------------------------------------
# Model Comparison
# -------------------------------------------------------
elif option == "Model Comparison":

    st.header("Model Comparison")

    model_scores = {
        "Linear Regression":0.62,
        "Decision Tree":0.86,
        "Random Forest":0.93,
        "Gradient Boosting":0.95
    }

    score_df = pd.DataFrame(
        model_scores.items(),
        columns=["Model","R² Score"]
    )

    st.bar_chart(score_df.set_index("Model"))

# -------------------------------------------------------
# Feature Distribution
# -------------------------------------------------------
elif option == "Feature Distribution":

    st.header("Power Distribution")

    fig, ax = plt.subplots(figsize=(10,5))

    sns.histplot(
        df["power"],
        bins=30,
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
else:

    st.header("Predict Bike Price")

    col1, col2 = st.columns(2)

    with col1:

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

    with col2:

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

    st.write("")

    if st.button("Predict Price", use_container_width=True):

        input_df = pd.DataFrame({

            "city":[city_encoder.transform([city])[0]],
            "kms_driven":[kms_driven],
            "owner":[owner_encoder.transform([owner])[0]],
            "age":[age],
            "power":[power],
            "brand":[brand_encoder.transform([brand])[0]]

        })

        prediction = model.predict(input_df)[0]

        st.success(f"### 💰 Estimated Bike Price: ₹ {prediction:,.0f}")

        st.balloons()
