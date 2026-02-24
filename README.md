📉 Telecom Customer Churn Prediction
Project Overview

This project demonstrates the use of machine learning to predict customer churn in the telecom industry.

Customer churn — when a customer cancels their service — is a critical metric for subscription-based businesses. Retaining existing customers is often far more cost-effective than acquiring new ones, making churn prediction essential for business growth and operational strategy.

This project covers the full analytics workflow:

Data exploration and visualization

Feature selection and preprocessing

Predictive modeling using Logistic Regression

Interactive risk prediction via a Streamlit app

Dataset

The dataset contains customer-level information from a telecom company, including account, usage, billing, and service interaction data. Each row represents a unique customer.

Feature	Description
Churn	Target variable (1 = customer cancelled, 0 = active)
AccountWeeks	Number of weeks customer has had an active account
ContractRenewal	1 if recently renewed contract, 0 otherwise
DataPlan	1 if customer has a data plan, 0 otherwise
DataUsage	Monthly data usage in GB
CustServCalls	Number of customer service calls
DayMins	Average daytime minutes per month
DayCalls	Average number of daytime calls
MonthlyCharge	Average monthly bill
OverageFee	Largest overage fee in the last 12 months
RoamMins	Average roaming minutes per month
Exploratory Data Analysis (EDA)

EDA focuses on identifying patterns that differentiate churned and retained customers:

Customers with higher customer service calls often churn, suggesting dissatisfaction or unresolved issues.

Contract renewals reduce churn risk, highlighting the importance of retention mechanisms.

Customers facing high monthly charges or overage fees are more likely to churn.

Lower engagement metrics (e.g., DataUsage and RoamMins) indicate lower perceived value and higher churn risk.

Modeling Approach

The project uses Logistic Regression for churn prediction:

Features are standardized using StandardScaler.

Stratified train-test split ensures balanced evaluation.

ROC-AUC is used to evaluate model performance.

Model outputs probability of churn, enabling risk segmentation.

Key Insights

High service call frequency → indicates potential dissatisfaction.

Lack of contract renewal → increases churn probability.

High monthly charges or overage fees → contributes to churn.

Lower usage levels → indicates customers may not perceive value.

Business Recommendations

Based on modeling results and feature insights, the following strategies can reduce churn:

Targeted engagement for high-risk customers

Optimized pricing plans for customers with high charges or overages

Customer support escalation for frequent callers

Contract renewal incentives

Monitoring usage patterns to identify disengaged customers early

Streamlit Application

An interactive Streamlit app demonstrates the project:

View the dataset and summary statistics

Explore churn patterns with charts

Train and evaluate the model

Simulate churn risk for individual customers

The app provides an end-to-end demonstration of data-driven decision support in a telecom business context.

Project Structure
customer_churn/
│
├── app.py               # Streamlit application
├── data/
│   └── churn.csv        # Telecom customer dataset
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
Dependencies

Python 3.10+

pandas

numpy

scikit-learn

joblib

streamlit

Install dependencies:

pip install -r requirements.txt

How to Run the App

Clone the repository

Install dependencies (pip install -r requirements.txt)

Run the Streamlit app:

streamlit run app.py

Use the sidebar to navigate between Dataset Overview, EDA, Model Training, and Churn Prediction.

Author

Janice Ivana – Designed, coded, and deployed the full Telecom Customer Churn Prediction workflow, including data preprocessing, modeling, and interactive deployment.
Website version: https://fintech-app.streamlit.app/