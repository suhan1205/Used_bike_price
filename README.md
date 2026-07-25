# 🎓 Student Exam Score Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project predicts a student's final exam score using an Artificial Neural Network (ANN). The model learns the relationship between academic, personal, and lifestyle factors to estimate the expected exam score.

The project demonstrates the complete deep learning workflow, including data preprocessing, feature engineering, model training, evaluation, and deployment with Streamlit.

---

## 🚀 Features

- Student exam score prediction
- Artificial Neural Network (ANN) Regression
- Data preprocessing and feature scaling
- Categorical feature encoding
- Model evaluation using regression metrics
- Interactive Streamlit web application

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Streamlit
- Joblib

---

## 📂 Project Structure

```
Student-Exam-Score-Prediction-ANN/
│
├── dataset/
│   └── student_performance.csv
│
├── models/
│   ├── ann_model.keras
│   ├── scaler.pkl
│   ├── encoder.pkl
│
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── images/
```

---

## 📊 Dataset Features

- Hours Studied
- Attendance
- Previous Scores
- Sleep Hours
- Tutoring Sessions
- Physical Activity
- Internet Access
- Motivation Level
- Family Income
- Teacher Quality
- School Type
- Distance from Home
- Gender

### 🎯 Target Variable

```
Exam Score
```

---

## 🧠 ANN Architecture

```
Input Layer
        │
Dense (128, ReLU)
        │
Batch Normalization
        │
Dropout (0.3)
        │
Dense (64, ReLU)
        │
Dropout (0.2)
        │
Dense (32, ReLU)
        │
Dense (1)
```

---

## 📈 Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/student-exam-score-ann.git
```

Move into the project directory

```bash
cd student-exam-score-ann
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 💻 Streamlit Application

The web application allows users to:

- Enter student details
- Predict the expected exam score
- View the predicted result instantly

---

## 📸 Application Screenshot

_Add your Streamlit application screenshot here._

---

## 🎯 Future Improvements

- Hyperparameter tuning
- Early stopping
- Cross-validation
- Feature importance analysis
- Improved user interface
- Model comparison with other regression algorithms

---

## 👨‍💻 Author

**Hardik Chaturvedi**

BCA Student | Machine Learning & Deep Learning Enthusiast

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub.
