# app_heart.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="❤️ Heart Disease Detector", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Detection")
st.caption("Enter patient details to estimate the probability of heart disease.")

# Load pipeline
MODEL_PATH = "heart_disease_model.pkl"
bundle = joblib.load(MODEL_PATH)
pipe = bundle["pipeline"]
feature_order = bundle["feature_order"]
numeric_cols = bundle["numeric_cols"]
categorical_cols = bundle["categorical_cols"]

st.subheader("Patient Inputs")

with st.form("heart_form"):
    inputs = {}

    # Numeric fields (typical ranges for guidance)
    if "age" in feature_order:
        inputs["age"] = st.number_input("Age (years)", min_value=1, max_value=110, value=54)
    if "trestbps" in feature_order:
        inputs["trestbps"] = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
    if "chol" in feature_order:
        inputs["chol"] = st.number_input("Serum Cholesterol (mg/dl)", min_value=80, max_value=700, value=246)
    if "thalach" in feature_order:
        inputs["thalach"] = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    if "oldpeak" in feature_order:
        inputs["oldpeak"] = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # Categorical / discrete
    # sex: 0=female, 1=male
    if "sex" in feature_order:
        inputs["sex"] = st.selectbox("Sex", options=[0, 1], index=1, help="0 = Female, 1 = Male")

    # cp: chest pain type (0..3)
    if "cp" in feature_order:
        inputs["cp"] = st.selectbox("Chest Pain Type (cp)", options=[0,1,2,3], index=0,
                                    help="0 = typical angina, 1 = atypical angina, 2 = non-anginal, 3 = asymptomatic")

    # fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    if "fbs" in feature_order:
        inputs["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0,1], index=0)

    # restecg: resting ECG (0..2)
    if "restecg" in feature_order:
        inputs["restecg"] = st.selectbox("Resting ECG (restecg)", options=[0,1,2], index=1)

    # exang: exercise induced angina (1 = yes; 0 = no)
    if "exang" in feature_order:
        inputs["exang"] = st.selectbox("Exercise Induced Angina (exang)", options=[0,1], index=0)

    # slope: slope of the peak exercise ST segment (0..2)
    if "slope" in feature_order:
        inputs["slope"] = st.selectbox("Slope of Peak Exercise ST (slope)", options=[0,1,2], index=1)

    # ca: number of major vessels colored by fluoroscopy (0..4)
    if "ca" in feature_order:
        inputs["ca"] = st.selectbox("Number of Major Vessels (ca)", options=[0,1,2,3,4], index=0)

    # thal: 0/1/2/3 depending on dataset version (commonly 1=normal,2=fixed defect,3=reversible defect)
    if "thal" in feature_order:
        inputs["thal"] = st.selectbox("Thalassemia (thal)", options=[0,1,2,3], index=2)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row DataFrame in the exact feature order used during training
    row = {col: inputs.get(col, 0) for col in feature_order}
    X_new = pd.DataFrame([row])

    proba = pipe.predict_proba(X_new)[0][1]
    pred = int(proba >= 0.5)  # simple threshold; you can tune this

    st.markdown("---")
    st.subheader("Prediction")
    if pred == 1:
        st.error(f"⚠️ **High risk of Heart Disease**\n\nEstimated probability: **{proba:.2%}**")
    else:
        st.success(f"✅ **Low risk of Heart Disease**\n\nEstimated probability: **{proba:.2%}**")

    st.caption("Note: This tool is for educational purposes and not a medical diagnosis.")
