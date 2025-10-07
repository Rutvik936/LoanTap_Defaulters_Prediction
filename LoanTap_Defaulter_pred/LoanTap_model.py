# app.py
import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load trained models
# ----------------------------
rf = joblib.load("rf_loanTap.pkl")    # Random Forest model
xgb = joblib.load("xgb_loanTap.pkl")  # XGBoost model

st.title("Loan Default Prediction")
st.write("Enter loan details to predict default risk:")

# ----------------------------
# Numeric features
# ----------------------------
loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
term = st.selectbox("Term", ['36 months','60 months'])
int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
installment = st.number_input("Monthly Installment", min_value=0.0, value=500.0)
emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
annual_inc = st.number_input("Annual Income", min_value=0, value=50000)
dti_capped = st.number_input("Debt to Income Ratio", min_value=0.0, value=15.0)
open_acc = st.number_input("Open Accounts", min_value=0, value=5)
total_acc = st.number_input("Total Accounts", min_value=0, value=10)
pub_rec = st.number_input("Public Records", min_value=0, value=0)
pub_rec_bankruptcies = st.number_input("Bankruptcies", min_value=0, value=0)
revol_bal = st.number_input("Revolving Balance", min_value=0, value=5000)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, value=50.0)
mort_acc = st.number_input("Mortgage Accounts", min_value=0, value=0)
credit_history_length_capped = st.number_input("Credit History Length (years)", min_value=0, value=10)
issue_year = st.number_input("Issue Year", min_value=2000, max_value=2030, value=2023)
issue_month = st.number_input("Issue Month", min_value=1, max_value=12, value=1)
issue_quarter = st.number_input("Issue Quarter", min_value=1, max_value=4, value=1)

# ----------------------------
# Categorical features
# ----------------------------
grade_options = ['A','B','C','D','E','F','G']
grade = st.selectbox("Grade", grade_options)

sub_grade_options = [f"{g}{i}" for g in grade_options for i in range(1,6)]
sub_grade = st.selectbox("Sub Grade", sub_grade_options)

home_options = ['RENT','MORTGAGE','OWN','OTHER']
home_ownership = st.selectbox("Home Ownership", home_options)

verification_options = ['Not Verified','Verified','Source Verified']
verification_status = st.selectbox("Verification Status", verification_options)

application_options = ['Individual','Joint']
application_type = st.selectbox("Application Type", application_options)

initial_list_options = ['w','f']
initial_list_status = st.selectbox("Initial List Status", initial_list_options)

purpose_options = [
    'debt_consolidation','credit_card','home_improvement','other',
    'major_purchase','small_business','car','medical','moving',
    'vacation','renewable_energy','house','educational'
]
purpose_grouped = st.selectbox("Purpose Grouped", purpose_options)

emp_experience_options = ['Junior','Mid','Senior','Executive']
emp_experience_level = st.selectbox("Employment Level", emp_experience_options)

# ----------------------------
# Prepare input DataFrame
# ----------------------------
input_dict = {
    'loan_amnt': loan_amnt,
    'term': 36 if term=='36 months' else 60,
    'int_rate': int_rate,
    'installment': installment,
    'emp_length': emp_length,
    'annual_inc': annual_inc,
    'dti_capped': dti_capped,
    'open_acc': open_acc,
    'total_acc': total_acc,
    'pub_rec': pub_rec,
    'pub_rec_bankruptcies': pub_rec_bankruptcies,
    'revol_bal': revol_bal,
    'revol_util': revol_util,
    'mort_acc': mort_acc,
    'credit_history_length_capped': credit_history_length_capped,
    'grade': grade,
    'sub_grade': sub_grade,
    'home_ownership': home_ownership,
    'verification_status': verification_status,
    'application_type': application_type,
    'initial_list_status': initial_list_status,
    'purpose_grouped': purpose_grouped,
    'emp_experience_level': emp_experience_level,
    'issue_year': issue_year,
    'issue_month': issue_month,
    'issue_quarter': issue_quarter
}

input_df = pd.DataFrame([input_dict])

# ----------------------------
# One-hot encode categorical features
# ----------------------------
cat_cols = [
    'grade','sub_grade','home_ownership','verification_status',
    'application_type','initial_list_status','purpose_grouped',
    'emp_experience_level'
]

input_df = pd.get_dummies(input_df, columns=cat_cols)

# ----------------------------
# Align input columns with training features
# ----------------------------
rf_features = [
    'loan_amnt','term','int_rate','installment','emp_length','annual_inc',
    'dti_capped','open_acc','total_acc','pub_rec','pub_rec_bankruptcies',
    'revol_bal','revol_util','mort_acc','credit_history_length_capped',
    'grade','sub_grade','home_ownership','verification_status',
    'application_type','initial_list_status','purpose_grouped',
    'emp_experience_level','issue_year','issue_month','issue_quarter'
]

# Add missing columns with 0
for col in rf_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[rf_features]

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    rf_pred = rf.predict(input_df)[0]
    rf_prob = rf.predict_proba(input_df)[0][1]

    xgb_pred = xgb.predict(input_df)[0]
    xgb_prob = xgb.predict_proba(input_df)[0][1]

   
     # Random Forest output
    rf_icon = "❌" if rf_pred == 1 else "✅"
    rf_text = f"Random Forest: {'Default' if rf_pred==1 else 'No Default'} ({rf_prob*100:.2f}%) {rf_icon}"
    st.markdown(f"<p style='font-size:16px'>{rf_text}</p>", unsafe_allow_html=True)

    # XGBoost output
    xgb_icon = "❌" if xgb_pred == 1 else "✅"
    xgb_text = f"XGBoost: {'Default' if xgb_pred==1 else 'No Default'} ({xgb_prob*100:.2f}%) {xgb_icon}"
    st.markdown(f"<p style='font-size:16px'>{xgb_text}</p>", unsafe_allow_html=True)
    
