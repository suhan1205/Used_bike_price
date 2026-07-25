# 🏍️ Used Bike Price Prediction

A Machine Learning web application built using **Python**, **Scikit-learn**, and **Streamlit** that predicts the selling price of a used bike based on its specifications.

## 🚀 Live Demo

Add your Streamlit link here:

https://your-streamlit-app-url.streamlit.app

---

## 📌 Features

- Predicts used bike prices instantly.
- User-friendly Streamlit interface.
- Machine Learning regression model.
- Supports different brands, cities, owners, bike age, engine power, and kilometers driven.
- Clean and responsive UI.

---

## 📂 Project Structure

```
Used_bike_price/
│
└── regression_streamlit/
    │── used_bikes_price_prediction.py
    │── bike_price_model.pkl
    │── label_encoders.pkl
    │── Used_Bikes.csv
    │── requirements.txt
    │── README.md
```

---

## 📊 Dataset

The model is trained on a dataset containing used bike listings with features such as:

- Brand
- City
- Kilometers Driven
- Owner
- Bike Age
- Engine Power (CC)
- Selling Price (Target)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 🤖 Machine Learning Model

The final model used for prediction is:

- **Gradient Boosting Regressor**

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Used_bike_price.git
```

Go to the project folder:

```bash
cd Used_bike_price/regression_streamlit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run used_bikes_price_prediction.py
```

---

## 📈 Input Features

The application accepts the following inputs:

- Brand
- City
- Kilometers Driven
- Bike Age
- Engine Power (CC)
- Owner

---

## 🎯 Output

The model predicts the estimated resale price of the selected used bike.

---

## 📷 Application Preview

(Add screenshots here)

---

## 📜 License

This project is created for educational and portfolio purposes.

---

## 👨‍💻 Author

**Hardik Chaturvedi**

BCA Student | Machine Learning Enthusiast

GitHub:
https://github.com/suhan1205
