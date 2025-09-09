import streamlit as st

# 🚨 MUST BE THE FIRST Streamlit command
st.set_page_config(page_title="Fraud Detection App", page_icon="🔍", layout="centered")

import pandas as pd
import numpy as np
import joblib

# ===============================
# Load Model, Scaler, and Feature Order
# ===============================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("stacking_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_order = joblib.load("feature_order.pkl")
        return model, scaler, feature_order
    except FileNotFoundError:
        st.error("❌ Model/scaler/feature files not found. Ensure they exist in your project folder.")
        st.stop()

model, scaler, feature_order = load_assets()

# ===============================
# UI Layout
# ===============================
st.title("🔍 Fraud Detection Prediction App")
st.markdown("Enter transaction details and click **Predict** to check if it's fraudulent.")

with st.container():
    st.header("📋 Transaction Details")

    transaction_type = st.selectbox(
        "Transaction Type",
        ('PAYMENT', 'CASH_OUT', 'DEBIT', 'CASH_IN', 'TRANSFER')
    )

    amount = st.number_input("Amount", value=0.0, step=100.0)
    oldbalanceOrg = st.number_input("Old Balance (Sender)", value=0.0, step=100.0)
    newbalanceOrig = st.number_input("New Balance (Sender)", value=0.0, step=100.0)
    oldbalanceDest = st.number_input("Old Balance (Recipient)", value=0.0, step=100.0)
    newbalanceDest = st.number_input("New Balance (Recipient)", value=0.0, step=100.0)

# --- Threshold ---
st.markdown("---")
st.subheader("⚙️ Prediction Settings")
prediction_threshold = st.slider(
    "Set Fraud Prediction Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

# ===============================
# Prediction Logic
# ===============================
if st.button("Predict"):
    # Create input dataframe with all required features
    input_data = pd.DataFrame(0, index=[0], columns=feature_order)

    # Fill numerical values
    if "amount" in input_data.columns:
        input_data["amount"] = amount
    if "oldbalanceOrg" in input_data.columns:
        input_data["oldbalanceOrg"] = oldbalanceOrg
    if "newbalanceOrig" in input_data.columns:
        input_data["newbalanceOrig"] = newbalanceOrig
    if "oldbalanceDest" in input_data.columns:
        input_data["oldbalanceDest"] = oldbalanceDest
    if "newbalanceDest" in input_data.columns:
        input_data["newbalanceDest"] = newbalanceDest

    # One-hot encode transaction type
    type_col = f"type_{transaction_type}"
    if type_col in input_data.columns:
        input_data[type_col] = 1

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction_proba = model.predict_proba(input_scaled)[:, 1][0]

    # Show result
    st.subheader("🔎 Prediction Result")
    if prediction_proba > prediction_threshold:
        st.error(f"🚨 This transaction is **FRAUDULENT** (Fraud Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ This transaction is **LEGITIMATE** (Legitimate Probability: {1 - prediction_proba:.2f})")

    st.write(f"Threshold used: **{prediction_threshold}**")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit for Fraud Detection")
