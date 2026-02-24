# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

import shap

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Telecom Customer Churn AI App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["About", "Dataset Overview", "EDA", "Modeling", "Churn Prediction"]
)

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/churn.csv")
    return df

df = load_data()
features = ["AccountWeeks","ContractRenewal","DataPlan","DataUsage",
            "CustServCalls","DayMins","DayCalls","MonthlyCharge",
            "OverageFee","RoamMins"]

# ---------------------------
# ABOUT PAGE
# ---------------------------
if page == "About":
    st.title("📉 Telecom Customer Churn Prediction")
    st.markdown("""
    Welcome! This app demonstrates how **AI and Machine Learning** can predict **customer churn** in the telecom industry.

    Retaining existing customers is more cost-effective than acquiring new ones. This app allows you to explore data, build predictive models, and simulate churn risk.

    ### Features
    - Explore customer data and statistics
    - Visualize churn patterns interactively
    - Train and evaluate multiple ML models
    - Predict churn for individual customers
    - AI-powered feature explanations using SHAP

    Navigate using the sidebar to explore Dataset, EDA, Modeling, and Churn Prediction.
    """)

# ---------------------------
# DATASET OVERVIEW PAGE
# ---------------------------
elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.dataframe(df.head(10))
    st.markdown("### Dataset Statistics")
    st.dataframe(df.describe())

# ---------------------------
# EXPLORATORY DATA ANALYSIS PAGE
# ---------------------------
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    
    st.subheader("Churn Distribution")
    fig = px.histogram(df, x="Churn", color="Churn",
                       labels={"Churn":"Churn"}, 
                       title="Distribution of Churn")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Customer Service Calls vs Churn")
    fig2 = px.box(df, x="Churn", y="CustServCalls", color="Churn",
                  title="Customer Service Calls by Churn")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Monthly Charges vs Churn")
    fig3 = px.box(df, x="Churn", y="MonthlyCharge", color="Churn",
                  title="Monthly Charge by Churn")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("Correlation Heatmap")
    corr = df[features + ["Churn"]].corr()
    fig4, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig4)

# ---------------------------
# MODELING PAGE
# ---------------------------
elif page == "Modeling":
    st.title("🤖 Model Training & Evaluation")
    
    # Model selection
    model_option = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
    
    # Train-test split
    X = df[features]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_pred_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:,1]
    
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    st.success(f"ROC-AUC Score: {roc_auc:.3f}")
    
    # Confusion matrix
    y_pred = (y_pred_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

# ---------------------------
# CHURN PREDICTION PAGE
# ---------------------------
elif page == "Churn Prediction":
    st.title("📌 Predict Churn for Individual Customers")
    
    st.markdown("Enter customer data below:")
    
    # Input widgets
    input_data = {}
    input_data["AccountWeeks"] = st.number_input("Account Weeks", min_value=0, max_value=1000, value=52)
    input_data["ContractRenewal"] = st.selectbox("Contract Renewal", [0,1])
    input_data["DataPlan"] = st.selectbox("Data Plan", [0,1])
    input_data["DataUsage"] = st.number_input("Monthly Data Usage (GB)", min_value=0.0, max_value=200.0, value=10.0)
    input_data["CustServCalls"] = st.number_input("Customer Service Calls", min_value=0, max_value=50, value=1)
    input_data["DayMins"] = st.number_input("Average Daytime Minutes", min_value=0.0, max_value=2000.0, value=200.0)
    input_data["DayCalls"] = st.number_input("Average Daytime Calls", min_value=0, max_value=500, value=50)
    input_data["MonthlyCharge"] = st.number_input("Monthly Charge ($)", min_value=0.0, max_value=1000.0, value=50.0)
    input_data["OverageFee"] = st.number_input("Largest Overage Fee ($)", min_value=0.0, max_value=500.0, value=0.0)
    input_data["RoamMins"] = st.number_input("Average Roaming Minutes", min_value=0.0, max_value=500.0, value=10.0)
    
    if st.button("Predict Churn"):
        input_df = pd.DataFrame([input_data])
        if   == "Logistic Regression":
            input_scaled = scaler.transform(input_df)
            prob = model.predict_proba(input_scaled)[:,1][0]
        else:
            prob = model.predict_proba(input_df)[:,1][0]
        
        st.subheader(f"Predicted Churn Probability: {prob:.2%}")
        
        # Risk category
        if prob >= 0.7:
            risk = "High"
        elif prob >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"
        st.info(f"Churn Risk Level: {risk}")
        
        # SHAP explanation (only for Random Forest)
        if model_option == "Random Forest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            st.subheader("Feature Contributions (SHAP Values)")
            shap.initjs()
            st_shap = st.pyplot(shap.force_plot(
                explainer.expected_value[1], shap_values[1], input_df, matplotlib=True
            ))