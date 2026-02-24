import streamlit as st
import pandas as pd
import joblib

st.title("FinTech Customer Churn Prediction")

df = pd.read_csv("data/churn.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Customer Churn Predictor")

monthly_charge = st.slider("Monthly Charge", 0, 200, 50)
cust_calls = st.slider("Customer Service Calls", 0, 10, 1)
overage_fee = st.slider("Overage Fee", 0, 200, 0)
roam_mins = st.slider("Roaming Minutes", 0, 500, 10)

if st.button("Predict Churn"):
    st.warning("High churn risk")  # placeholder