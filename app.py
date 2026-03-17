import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Telecom Churn AI",
    page_icon="📞",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Data & Model Logic
# ---------------------------
@st.cache_data
def load_data():
    try:
        # Using a sample path - ensure your CSV is here
        df = pd.read_csv("data/churn.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'data/churn.csv' exists.")
        return pd.DataFrame()

df = load_data()
features = ["AccountWeeks","ContractRenewal","DataPlan","DataUsage",
            "CustServCalls","DayMins","DayCalls","MonthlyCharge",
            "OverageFee","RoamMins"]

@st.cache_resource
def train_model(model_type, use_poly):
    if df.empty: return None, None, None, None, None
    
    X = df[features]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if model_type == "Logistic Regression":
        steps = [('scaler', StandardScaler())]
        if use_poly:
            steps.insert(0, ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
        steps.append(('lr', LogisticRegression(max_iter=1000, class_weight="balanced")))
    else:
        steps = [('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))]
        
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2058/2058092.png", width=100)
st.sidebar.title("Churn Control Center")
page = st.sidebar.selectbox("Select Workspace", ["Overview", "Exploration", "Model Performance", "Predictor"])

st.sidebar.divider()
model_option = st.sidebar.radio("ML Engine", ["Logistic Regression", "Random Forest"])
use_poly = st.sidebar.checkbox("Poly Features", value=False) if model_option == "Logistic Regression" else False

# Load Model
model, X_train, X_test, y_train, y_test = train_model(model_option, use_poly)

# ---------------------------
# Workspace: Overview
# ---------------------------
if page == "Overview":
    st.title("📞 Telecom Churn Intelligence")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{(df['Churn'].mean()*100):.1f}%")
    col3.metric("Avg Monthly Bill", f"${df['MonthlyCharge'].mean():.2f}")
    col4.metric("Avg Support Calls", round(df['CustServCalls'].mean(), 1))
    
    st.markdown("---")
    st.subheader("Recent Customer Log")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------------------
# Workspace: Exploration
# ---------------------------
elif page == "Exploration":
    st.title("🔍 Data Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(df, x="MonthlyCharge", color="Churn", marginal="box", 
                                 title="Monthly Charge Distribution", barmode="overlay",
                                 color_discrete_sequence=["#636EFA", "#EF553B"])
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_scatter = px.scatter(df, x="DayMins", y="MonthlyCharge", color="Churn",
                                 title="Usage vs. Billing", opacity=0.5)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Feature Correlation")
    corr = df[features + ["Churn"]].corr()
    fig_corr, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(corr, annot=True, cmap="RdBu", center=0, ax=ax)
    st.pyplot(fig_corr)

# ---------------------------
# Workspace: Model Performance
# ---------------------------
elif page == "Model Performance":
    st.title("🤖 Model Evaluation")
    
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_val = roc_auc_score(y_test, y_probs)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("ROC-AUC Score", f"{auc_val:.3f}")
        st.write("**Analysis:**")
        if auc_val > 0.8:
            st.success("Strong predictive power!")
        else:
            st.warning("Model needs more tuning.")

    with col2:
        fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={auc_val:.2f})",
                          labels={'x':'False Positive Rate', 'y':'True Positive Rate'})
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)

# ---------------------------
# Workspace: Predictor
# ---------------------------
elif page == "Predictor":
    st.title("🎯 Real-time Churn Prediction")
    
    with st.expander("Adjust Customer Profile", expanded=True):
        c1, c2, c3 = st.columns(3)
        input_data = {
            "AccountWeeks": c1.number_input("Account Age (Weeks)", 1, 500, 50),
            "ContractRenewal": c2.selectbox("Contract Renewed?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No"),
            "DataPlan": c3.selectbox("Data Plan?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No"),
            "DataUsage": c1.slider("Data Usage (GB)", 0.0, 50.0, 2.0),
            "CustServCalls": c2.slider("Service Calls", 0, 10, 1),
            "DayMins": c3.slider("Day Minutes", 0.0, 500.0, 180.0),
            "DayCalls": c1.slider("Day Calls", 1, 200, 100),
            "MonthlyCharge": c2.slider("Monthly Charge ($)", 10.0, 150.0, 50.0),
            "OverageFee": c3.slider("Overage Fee ($)", 0.0, 100.0, 5.0),
            "RoamMins": c1.slider("Roaming Mins", 0.0, 50.0, 10.0)
        }

    input_df = pd.DataFrame([input_data])
    prob = model.predict_proba(input_df)[:, 1][0]
    
    # Visual Gauge for Probability
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        title = {'text': "Churn Probability %"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}]
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Simple Feature Contribution (Difference from mean)
    st.subheader("💡 Why this score?")
    mean_values = df[features].mean()
    high_impact = []
    if input_data['CustServCalls'] > mean_values['CustServCalls']:
        high_impact.append("High number of service calls is driving risk up.")
    if input_data['ContractRenewal'] == 0:
        high_impact.append("Lack of contract renewal is a major churn indicator.")
    
    if high_impact:
        for note in high_impact:
            st.info(note)
    else:
        st.write("Customer behavior aligns with retention patterns.")