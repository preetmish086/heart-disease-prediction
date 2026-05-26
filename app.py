import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Cardio-Predict",
    page_icon="🫀",
    layout="centered"
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------- TITLE ---------------- #

st.title("🫀 Cardio-Predict ")
st.markdown(
    "Predict the likelihood of heart disease using Machine Learning."
)

st.divider()

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("ℹ️ About")

st.sidebar.success("✅ Model Accuracy: 94%")

st.sidebar.info(
    """
This ML model predicts the probability of heart disease based on medical attributes.

### Features Used
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- Slope
- Major Vessels
- Thalassemia

### Dataset
Heart Disease Dataset from Kaggle
"""
)

# ---------------- INPUT SECTION ---------------- #

st.subheader("🩺 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100)

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    cp = st.slider(
        "Chest Pain Type",
        0, 3
    )

    trestbps = st.number_input(
        "Resting Blood Pressure"
    )

    chol = st.number_input(
        "Cholesterol Level"
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    restecg = st.slider(
        "Rest ECG",
        0, 2
    )

with col2:

    thalach = st.number_input(
        "Maximum Heart Rate"
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0
    )

    slope = st.slider(
        "Slope",
        0, 2
    )

    ca = st.slider(
        "Number of Major Vessels",
        0, 4
    )

    thal = st.slider(
        "Thal",
        0, 3
    )

# ---------------- DATA PREPROCESSING ---------------- #

if sex == "Male":
    sex = 1 
else:
    sex = 0

# ---------------- PREDICTION ---------------- #

st.divider()

if st.button("🔍 Predict Heart Disease Risk"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(
            "⚠️ High chance of Heart Disease detected."
        )

    else:
        st.success(
            "✅ Low chance of Heart Disease."
        )

st.divider()

st.caption(
    "Made using Streamlit and Scikit-learn"
)