import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --------------------------------------------------
# Application Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    layout="wide"
)

st.title("📉 Telecom Customer Churn Prediction")
st.markdown(
    """
    This application predicts **customer churn** in a telecom context using
    customer tenure, usage behavior, billing information, and service interaction data.

    The goal is to identify **customers at risk of leaving** and support
    data-driven retention strategies.
    """
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    """
    Load the telecom churn dataset.
    Caching is used to improve app performance.
    """
    return pd.read_csv("data/churn.csv")

df = load_data()

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "Exploratory Data Analysis",
        "Model Training",
        "Churn Risk Prediction"
    ]
)

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------
if page == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Statistics")
    st.write(df.describe())

    churn_rate = df["Churn"].mean() * 100
    st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

# --------------------------------------------------
# Exploratory Data Analysis
# --------------------------------------------------
elif page == "Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

    with col2:
        st.markdown("### Average Customer Service Calls by Churn")
        st.bar_chart(df.groupby("Churn")["CustServCalls"].mean())

    st.markdown("### Average Monthly Charge by Churn Status")
    st.bar_chart(df.groupby("Churn")["MonthlyCharge"].mean())

    st.info(
        "Customers who churn tend to have **higher service call frequency** "
        "and **higher billing exposure**, indicating dissatisfaction and pricing friction."
    )

# --------------------------------------------------
# Model Training
# --------------------------------------------------
elif page == "Model Training":
    st.subheader("Model Training")

    # Feature selection based on business relevance
    feature_columns = [
        "AccountWeeks",
        "ContractRenewal",
        "DataPlan",
        "DataUsage",
        "CustServCalls",
        "DayMins",
        "DayCalls",
        "MonthlyCharge",
        "OverageFee",
        "RoamMins"
    ]

    X = df[feature_columns]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Pipeline ensures consistent preprocessing and prediction
    churn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    if st.button("Train Model"):
        churn_pipeline.fit(X_train, y_train)

        y_pred_prob = churn_pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        joblib.dump(churn_pipeline, "churn_model.pkl")

        st.success("Model trained and saved successfully.")
        st.metric("ROC-AUC Score", f"{roc_auc:.3f}")

        st.caption(
            "ROC-AUC is used to evaluate how well the model ranks churn risk, "
            "which is more informative than accuracy for imbalanced datasets."
        )

# --------------------------------------------------
# Churn Risk Prediction
# --------------------------------------------------
elif page == "Churn Risk Prediction":
    st.subheader("Predict Customer Churn Risk")

    if not os.path.exists("churn_model.pkl"):
        st.warning("Please train the model before making predictions.")
    else:
        model = joblib.load("churn_model.pkl")

        col1, col2 = st.columns(2)

        with col1:
            account_weeks = st.number_input("Account Tenure (Weeks)", 0, 1000, 100)
            contract_renewal = st.selectbox("Contract Renewal", [0, 1])
            data_plan = st.selectbox("Data Plan", [0, 1])
            data_usage = st.number_input("Monthly Data Usage (GB)", 0.0, 100.0, 5.0)
            cust_serv_calls = st.number_input("Customer Service Calls", 0, 20, 1)

        with col2:
            day_mins = st.number_input("Daytime Minutes", 0.0, 1000.0, 300.0)
            day_calls = st.number_input("Daytime Calls", 0, 300, 100)
            monthly_charge = st.number_input("Monthly Charge", 0.0, 500.0, 70.0)
            overage_fee = st.number_input("Overage Fee", 0.0, 500.0, 0.0)
            roam_mins = st.number_input("Roaming Minutes", 0.0, 500.0, 10.0)

        if st.button("Predict Churn"):
            input_df = pd.DataFrame([[
                account_weeks,
                contract_renewal,
                data_plan,
                data_usage,
                cust_serv_calls,
                day_mins,
                day_calls,
                monthly_charge,
                overage_fee,
                roam_mins
            ]], columns=[
                "AccountWeeks",
                "ContractRenewal",
                "DataPlan",
                "DataUsage",
                "CustServCalls",
                "DayMins",
                "DayCalls",
                "MonthlyCharge",
                "OverageFee",
                "RoamMins"
            ])

            churn_probability = model.predict_proba(input_df)[0][1]

            if churn_probability >= 0.70:
                st.error(f"High Churn Risk 🔴 ({churn_probability:.2%})")
            elif churn_probability >= 0.40:
                st.warning(f"Medium Churn Risk 🟠 ({churn_probability:.2%})")
            else:
                st.success(f"Low Churn Risk 🟢 ({churn_probability:.2%})")