# ❤️ Heart Disease Risk Prediction Dashboard

A machine learning powered web application that predicts the risk of heart disease using patient health data and a stacked ensemble learning model.

## 📌 Project Overview

This project is built using:

* **Python**
* **Streamlit** for the web interface
* **Machine Learning** for prediction
* **Stacked Ensemble Learning** model

The application allows users to enter patient medical information and receive:

* Heart disease risk prediction
* Probability score
* Risk visualization chart
* Model details

---

# 🚀 Features

✅ User-friendly Streamlit dashboard
✅ Real-time heart disease prediction
✅ Probability-based risk assessment
✅ Data preprocessing and scaling
✅ One-hot encoding support
✅ Visualization using Matplotlib
✅ Ensemble learning architecture

---

# 🧠 Machine Learning Model

The prediction system uses a **Stacked Ensemble Learning** approach.

## Base Models

* Random Forest
* Support Vector Machine (SVM)
* XGBoost

## Final Estimator

* Logistic Regression

## Problem Type

* Binary Classification

---

# 📂 Project Structure

```bash
Heart-Disease-Prediction/
│
├── app.py
├── heart_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📊 Input Features

The model uses the following patient information:

* Age
* Sex
* Location
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG Results
* Maximum Heart Rate
* Exercise Induced Angina
* Oldpeak
* ST Slope
* Thalassemia

---

# 📈 Output

The system predicts:

* HIGH RISK of Heart Disease
* LOW RISK of Heart Disease

It also displays:

* Risk probability score
* Visualization chart

---

# 🖥️ Technologies Used

* Python
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* Matplotlib
* Joblib

---

# 📷 Dashboard Preview

Add screenshots of your application here.

---

# 🔮 Future Improvements

* Add patient report download
* Improve UI/UX design
* Add more medical parameters
* Deploy using Streamlit Cloud
* Add authentication system
* Connect with real-time hospital database

---

# 👩‍💻 Author

Developed by Divya.

---

# 📜 License

This project is for educational and learning purposes.
