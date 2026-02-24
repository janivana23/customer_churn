# 📉 Telecom Customer Churn Prediction
## Project Overview

This project demonstrates the use of machine learning to predict customer churn in the telecom industry.

Customer churn — when a customer cancels their service — is a critical metric for subscription-based businesses. Retaining existing customers is often far more cost-effective than acquiring new ones, making churn prediction essential for business growth and operational strategy.

This project covers the full analytics workflow:

1. Data exploration and visualization

2. Feature selection and preprocessing

3. Predictive modeling using Logistic Regression

4. Interactive risk prediction via a Streamlit app

## Dataset

The dataset contains customer-level information from a telecom company, including account, usage, billing, and service interaction data. Each row represents a unique customer.

Feature	Description
- Churn	Target variable (1 = customer cancelled, 0 = active)
- AccountWeeks	Number of weeks customer has had an active account
- ContractRenewal	1 if recently renewed contract, 0 otherwise
- DataPlan	1 if customer has a data plan, 0 otherwise
- DataUsage	Monthly data usage in GB
- CustServCalls	Number of customer service calls
- DayMins	Average daytime minutes per month
- DayCalls	Average number of daytime calls
- MonthlyCharge	Average monthly bill
- OverageFee	Largest overage fee in the last 12 months
- RoamMins	Average roaming minutes per month

## Exploratory Data Analysis (EDA)

EDA focuses on identifying patterns that differentiate churned and retained customers:

- Customers with higher customer service calls often churn, suggesting dissatisfaction or unresolved issues.
- Contract renewals reduce churn risk, highlighting the importance of retention mechanisms.
- Customers facing high monthly charges or overage fees are more likely to churn.
- Lower engagement metrics (e.g., DataUsage and RoamMins) indicate lower perceived value and higher churn risk.

## Modeling Approach

The project uses Logistic Regression for churn prediction:

- Features are standardized using StandardScaler.
- Stratified train-test split ensures balanced evaluation.
- ROC-AUC is used to evaluate model performance.
- Model outputs probability of churn, enabling risk segmentation.

## Key Insights

- High service call frequency → indicates potential dissatisfaction.
- Lack of contract renewal → increases churn probability.
- High monthly charges or overage fees → contributes to churn.
- Lower usage levels → indicates customers may not perceive value.

## 🤖 AI What-If Scenario

This app includes an **AI-powered What-If feature** that simulates changes in customer behavior to reduce churn risk. 

- **Interactive Simulation:** Adjust customer attributes using sliders and immediately see how churn probability changes.  
- **Actionable Recommendations:** The app automatically suggests minimal adjustments that could lower the risk, such as:
  - Reducing the number of customer service calls  
  - Renewing a contract if not already renewed  
  - Optimizing monthly charges or overage fees  
  - Increasing engagement metrics like data usage or roaming minutes  
- **Dynamic Feedback:** The AI provides real-time probability updates and practical insights to help decision-making.  

This feature demonstrates how predictive analytics can guide customer retention strategies in a telecom context, offering **data-driven, actionable recommendations** beyond just churn prediction.

## Business Recommendations

Based on modeling results and feature insights, the following strategies can reduce churn:

1. Targeted engagement for high-risk customers
2. Optimized pricing plans for customers with high charges or overages
3. Customer support escalation for frequent callers
4. Contract renewal incentives
5. Monitoring usage patterns to identify disengaged customers early

## Streamlit Application

An interactive Streamlit app demonstrates the project:

- View the dataset and summary statistics
- Explore churn patterns with charts
- Train and evaluate the model
- Simulate churn risk for individual customers

The app provides an end-to-end demonstration of data-driven decision support in a telecom business context.

## Project Structure
customer_churn/
- app.py               # Streamlit application
- data/
  - churn.csv        # Telecom customer dataset
- requirements.txt     # Python dependencies
- README.md            # Project documentation

## How to Run the App

1. Clone the repository

2. Install dependencies (pip install -r requirements.txt)

3. Run the Streamlit app:

    streamlit run app.py

4. Use the sidebar to navigate between Dataset Overview, EDA, Model Training, and Churn Prediction.

5. Website: https://fintech-app.streamlit.app/

## Author

Janice Ivana – Designed, coded, and deployed the full Telecom Customer Churn Prediction workflow, including data preprocessing, modeling, and interactive deployment.
Website version: https://fintech-app.streamlit.app/