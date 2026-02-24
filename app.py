# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Telecom Customer Churn AI App",
    layout="wide",
    initial_sidebar_state="expanded"
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
# Sidebar Navigation & Model Selection
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["About", "Dataset Overview", "EDA", "Modeling", "Churn Prediction"]
)

# Global model selection
model_option = st.sidebar.selectbox(
    "Select Model for Prediction",
    ["Logistic Regression", "Random Forest"]
)

# Polynomial feature toggle for LR
use_poly = False
if model_option == "Logistic Regression":
    use_poly = st.sidebar.checkbox("Use Polynomial Features (interactions) for LR", value=False)

# ---------------------------
# Train & Cache Model
# ---------------------------
@st.cache_resource
def get_trained_model(model_option, use_poly=False):
    X = df[features]
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    if model_option == "Logistic Regression":
        if use_poly:
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(max_iter=2000, class_weight="balanced"))
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(max_iter=2000, class_weight="balanced"))
            ])
        pipeline.fit(X_train, y_train)
    else:
        pipeline = Pipeline([
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
    
    return pipeline, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = get_trained_model(model_option, use_poly)

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
    - AI-powered what-if scenarios for churn reduction
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
    st.plotly_chart(fig3)
    
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
    
    # Predict probabilities
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
    
    # Feature importance for Random Forest
    if model_option == "Random Forest":
        st.subheader("Feature Importance")
        rf_model = model.named_steps["rf"]
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(importance.set_index("Feature"))

# ---------------------------
# CHURN PREDICTION PAGE (Live Simulator + What-If AI)
# ---------------------------
elif page == "Churn Prediction":
    st.title("📌 Predict Churn for Individual Customers (Live Simulator)")
    st.markdown("Adjust the customer attributes using sliders and see how churn probability changes in real-time.")

    # Sliders / inputs
    input_data = {
        "AccountWeeks": st.slider("Account Weeks", 0, 1000, 52),
        "ContractRenewal": st.select_slider("Contract Renewal", options=[0,1], value=1),
        "DataPlan": st.select_slider("Data Plan", options=[0,1], value=1),
        "DataUsage": st.slider("Monthly Data Usage (GB)", 0.0, 200.0, 10.0),
        "CustServCalls": st.slider("Customer Service Calls", 0, 50, 1),
        "DayMins": st.slider("Average Daytime Minutes", 0.0, 2000.0, 200.0),
        "DayCalls": st.slider("Average Daytime Calls", 0, 500, 50),
        "MonthlyCharge": st.slider("Monthly Charge ($)", 0.0, 1000.0, 50.0),
        "OverageFee": st.slider("Largest Overage Fee ($)", 0.0, 500.0, 0.0),
        "RoamMins": st.slider("Average Roaming Minutes", 0.0, 500.0, 10.0)
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict churn probability
    prob = model.predict_proba(input_df)[:,1][0]
    st.subheader(f"Predicted Churn Probability: {prob:.2%}")

    # Risk category
    if prob >= 0.7:
        st.error("Churn Risk Level: High")
    elif prob >= 0.4:
        st.warning("Churn Risk Level: Medium")
    else:
        st.success("Churn Risk Level: Low")

    # ---------------------------
    # AI What-If Recommendations
    # ---------------------------
    st.subheader("🤖 AI What-If Recommendations")
    st.markdown("""
    This simple AI engine suggests **which features to adjust** to reduce churn probability.
    """)

    suggestions = []

    # Example rules-based AI: adjust features based on general churn trends
    if input_data["CustServCalls"] > 2:
        suggestions.append("- Reduce customer service calls or resolve issues promptly")
    if input_data["ContractRenewal"] == 0:
        suggestions.append("- Encourage contract renewal to increase retention")
    if input_data["MonthlyCharge"] > 70:
        suggestions.append("- Consider offering discounts or plan optimization")
    if input_data["OverageFee"] > 20:
        suggestions.append("- Monitor and reduce overage fees")
    if input_data["DataUsage"] < 5:
        suggestions.append("- Encourage higher data plan engagement")
    if input_data["RoamMins"] < 10:
        suggestions.append("- Provide roaming incentives to increase value perception")

    if suggestions:
        for s in suggestions:
            st.info(s)
    else:
        st.success("Customer attributes look good! Low churn risk factors detected.")