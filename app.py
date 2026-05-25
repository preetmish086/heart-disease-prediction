import streamlit as st
import numpy as np
import joblib 

model=joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Heart Disease Prediction App")

st.sidebar.title("About")
st.sidebar.success("Model Accuracy : 94%")
st.sidebar.info(
    """This AI model predicts the likelihood of heart disease using machine learning.  
    It takes input of the following attributes-  
    Attribute Information:  
    -------------------------  
    age-age  
    sex- sex  
    cp- chest pain type (4 values)  
    trestbps- resting blood pressure  
    chol- serum cholestoral in mg/dl  
    fbs- fasting blood sugar > 120 mg/dl  
    restecg- resting electrocardiographic results (values 0,1,2)  
    thalach- maximum heart rate achieved  
    exang- exercise induced angina  
    oldpeak = ST depression induced by exercise relative to rest  
    slope- the slope of the peak exercise ST segment  
    ca- number of major vessels (0-3) colored by flourosopy  
    thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
    target- 0 = low risk of heart disease, 1 = high risk of heart disease  
    --------------------------  
    Dataset used- Heart Disease Dataset  
    Source- Kaggle
    """
)

age = st.number_input("Age", min_value=1, max_value=100)

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.slider("Chest Pain Type", 0, 3)

trestbps = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])

restecg = st.slider("Rest ECG", 0, 2)

thalach = st.number_input("Maximum Heart Rate")

exang = st.selectbox("Exercise Induced Angina", [0,1])

oldpeak = st.number_input("Oldpeak")

slope = st.slider("Slope", 0, 2)

ca = st.slider("Number of Major Vessels", 0, 4)

thal = st.slider("Thal", 0, 3)

if sex == "Male":
    sex = 1
else:
    sex = 0

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach,
                            exang, oldpeak, slope,
                            ca, thal]])
    input_data = scaler.transform(input_data)
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High chance of Heart Disease")
    else:
        st.success("Low chance of Heart Disease")
