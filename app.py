import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="FinTech Customer Churn Prediction",
    layout="wide"
)

st.title("📉 FinTech Customer Churn Prediction")
st.write(
    "This application predicts customer churn using behavioral, usage, "
    "and billing data from a subscription-based FinTech service."
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/churn.csv")

df = load_data()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "EDA", "Train Model", "Churn Prediction"]
)

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------
if page == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe())

    churn_rate = df["Churn"].mean() * 100
    st.metric("Overall Churn Rate (%)", f"{churn_rate:.2f}")

# --------------------------------------------------
# EDA
# --------------------------------------------------
elif page == "EDA":
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

    with col2:
        st.write("### Customer Service Calls vs Churn")
        st.bar_chart(df.groupby("Churn")["CustServCalls"].mean())

    st.write("### Average Charges by Churn Status")
    st.bar_chart(df.groupby("Churn")["MonthlyCharge"].mean())

# --------------------------------------------------
# Train Model
# --------------------------------------------------
elif page == "Train Model":
    st.subheader("Model Training")

    features = [
        "AccountWeeks", "ContractRenewal", "DataPlan",
        "DataUsage", "CustServCalls", "DayMins",
        "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins"
    ]

    X = df[features]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    if st.button("Train Churn Model"):
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        joblib.dump(pipeline, "churn_model.pkl")

        st.success("Model trained and saved successfully!")
        st.metric("ROC-AUC Score", f"{auc:.3f}")

# --------------------------------------------------
# Churn Prediction
# --------------------------------------------------
elif page == "Churn Prediction":
    st.subheader("Predict Customer Churn")

    if not os.path.exists("churn_model.pkl"):
        st.warning("Please train the model first.")
    else:
        model = joblib.load("churn_model.pkl")

        col1, col2 = st.columns(2)

        with col1:
            account_weeks = st.number_input("Account Weeks", 0, 1000, 100)
            contract_renewal = st.selectbox("Contract Renewal", [0, 1])
            data_plan = st.selectbox("Data Plan", [0, 1])
            data_usage = st.number_input("Monthly Data Usage (GB)", 0.0, 100.0, 5.0)
            cust_calls = st.number_input("Customer Service Calls", 0, 20, 1)

        with col2:
            day_mins = st.number_input("Daytime Minutes", 0.0, 1000.0, 300.0)
            day_calls = st.number_input("Daytime Calls", 0, 300, 100)
            monthly_charge = st.number_input("Monthly Charge", 0.0, 500.0, 70.0)
            overage_fee = st.number_input("Overage Fee", 0.0, 500.0, 0.0)
            roam_mins = st.number_input("Roaming Minutes", 0.0, 500.0, 10.0)

        if st.button("Predict Churn Risk"):
            input_data = pd.DataFrame([[
                account_weeks, contract_renewal, data_plan,
                data_usage, cust_calls, day_mins,
                day_calls, monthly_charge, overage_fee, roam_mins
            ]], columns=[
                "AccountWeeks", "ContractRenewal", "DataPlan",
                "DataUsage", "CustServCalls", "DayMins",
                "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins"
            ])

            churn_prob = model.predict_proba(input_data)[0][1]

            if churn_prob >= 0.7:
                st.error(f"High Churn Risk 🔴 ({churn_prob:.2%})")
            elif churn_prob >= 0.4:
                st.warning(f"Medium Churn Risk 🟠 ({churn_prob:.2%})")
            else:
                st.success(f"Low Churn Risk 🟢 ({churn_prob:.2%})")