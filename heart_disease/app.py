import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="Heart Disease Risk Analyzer",
                   page_icon="❤️",
                   layout="wide")

# ---------------- LOAD MODEL FILES ----------------

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---------------- SIDEBAR INPUT ----------------

st.sidebar.title("🩺 Patient Information")

age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
location = st.sidebar.selectbox("Location",
                                ["Hungary", "Switzerland", "VA Long Beach"])
chest_pain = st.sidebar.selectbox("Chest Pain Type",
                                  ["typical angina",
                                   "atypical angina",
                                   "non-anginal pain"])
rest_bp = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", ["True", "False"])
rest_ecg = st.sidebar.selectbox("Resting ECG",
                                ["normal", "ST-T abnormality","lv hypertrophy"])
max_hr = st.sidebar.number_input("Maximum Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["True", "False"])
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.sidebar.selectbox("ST Slope", ["upsloping", "flat","Downsloping"])
thal = st.sidebar.selectbox("Thalassemia",
                            ["normal", "reversable defect","Fixed defect"])

# ---------------- MAIN TITLE ----------------

st.title("❤️ Heart Disease Risk Prediction Dashboard")
st.markdown("Advanced Stacked Ensemble Learning Model")

# ---------------- PREDICTION ----------------

if st.button("🔍 Analyze Risk"):

    # Create input dictionary AFTER collecting inputs
    input_dict = {
        "age": age,
        "Resting_BP": rest_bp,
        "Cholesterol": chol,
        "Maximum Heart Rate Achieved": max_hr,
        "oldpeak": oldpeak,
        "sex": sex,
        "Location": location,
        "Chest_pain_type": chest_pain,
        "Fasting Blood Sugar": fbs,
        "Resting ECG Results": rest_ecg,
        "Exercise-Induced Angina": exang,
        "slope": slope,
        "Thalassemia": thal
    }

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Apply same encoding
    input_df = pd.get_dummies(input_df)

    # Align with training features
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ---------------- RESULT SECTION ----------------

    st.subheader("🧾 Risk Assessment Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠ HIGH RISK of Heart Disease")
        else:
            st.success("✅ LOW RISK of Heart Disease")

        st.progress(int(probability * 100))
        st.write(f"Risk Probability: **{probability:.2f}**")

    # -------- Probability Chart --------
    with col2:
        fig, ax = plt.subplots()
        ax.bar(["Risk Level"], [probability])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Risk Probability Gauge")
        st.pyplot(fig)

    st.markdown("---")

    # -------- Model Info --------
    st.subheader("🧠 Model Information")
    st.write("""
    - Base Models: Random Forest  
    - SVM  
    - XGBoost  
    - Final Estimator: Logistic Regression  
    - Type: Binary Classification  
    - Method: Stacked Ensemble Learning  
    """)