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

    st.header("📊 Correlation Heatmap")

    numeric_df = df[["price", "kms_driven", "age", "power"]]

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="rocket",
        linewidths=1,
        fmt=".2f",
        cbar=True,
        square=True,
        ax=ax
    )

    ax.set_title("Correlation Heatmap", fontsize=18)

    st.pyplot(fig)

    st.info("""
### 📌 Insights

- **Power** has the strongest positive correlation with **Price**.
- **Bike Age** has a weak negative correlation with **Price**.
- **Kilometers Driven** also has a weak negative relationship with **Price**.
- As engine power increases, the resale price generally increases.
""")
# -------------------------------------------------------
# Model Comparison
# -------------------------------------------------------
elif option == "Model Comparison":

    st.header("📈 Machine Learning Model Comparison")

    model_scores = {
        "Linear Regression": 0.717264,
        "Decision Tree Regressor": 0.755922,
        "Random Forest Regressor": 0.875684,
        "Support Vector Regressor": 0.004984,
        "K-Nearest Neighbors": 0.663884,
        "Gradient Boosting Regressor": 0.900724
    }

    score_df = pd.DataFrame(
        model_scores.items(),
        columns=["Model", "R² Score"]
    )

    st.dataframe(score_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        score_df["Model"],
        score_df["R² Score"]
    )

    ax.set_title("Regression Model Performance")
    ax.set_ylabel("R² Score")
    plt.xticks(rotation=20, ha="right")

    # Show values on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            fontsize=9
        )

    st.pyplot(fig)

    best_model = score_df.loc[score_df["R² Score"].idxmax()]

    st.success(
        f"🏆 Best Model: **{best_model['Model']}** "
        f"(R² Score = **{best_model['R² Score']:.6f}**)"
    )

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
