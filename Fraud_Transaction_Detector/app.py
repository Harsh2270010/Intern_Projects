# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load("model.pkl")

st.title("💳 Fraud Detection App")
st.write("Enter transaction details to check if it is fraud or not.")

# Identify numeric and categorical columns from the pipeline
NUMERIC = model.named_steps['pre'].transformers_[0][2]
CATEGORICAL = []
if len(model.named_steps['pre'].transformers_) > 1:
    CATEGORICAL = model.named_steps['pre'].transformers_[1][2]

# Input form
with st.form("fraud_form"):
    input_data = {}

    for col in NUMERIC:
        input_data[col] = st.number_input(col, value=0.0, step=0.01)

    for col in CATEGORICAL:
        input_data[col] = st.text_input(col, value="")

    submitted = st.form_submit_button("Predict Fraud")

# Prediction
if submitted:
    df_input = pd.DataFrame([input_data])

    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud detected! Probability: {probability:.2%}")
    else:
        st.success(f"✅ Not Fraud. Probability of fraud: {probability:.2%}")
