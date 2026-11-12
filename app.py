import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Home Loan Default Risk Predictor", page_icon="🏦", layout="centered")

# Load trained LightGBM model
model = joblib.load('loan_model.pkl')

st.title("🏦 Home Loan Eligibility & Default Risk Predictor")
st.markdown("Evaluate applicant creditworthiness using our **LightGBM** model trained on Home Credit relational data.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income ($)", min_value=10000.0, value=150000.0, step=5000.0)
    credit_amt = st.number_input("Requested Loan Amount ($)", min_value=10000.0, value=500000.0, step=10000.0)
    annuity = st.number_input("Loan Annuity / Yearly Payment ($)", min_value=1000.0, value=25000.0, step=1000.0)
    age = st.slider("Applicant Age (Years)", min_value=20, max_value=70, value=35)
    emp_years = st.slider("Years Employed", min_value=0, max_value=45, value=5)

with col2:
    ext_1 = st.slider("External Credit Score 1 (0.0 - 1.0)", 0.0, 1.0, 0.55, 0.01)
    ext_2 = st.slider("External Credit Score 2 (0.0 - 1.0)", 0.0, 1.0, 0.60, 0.01)
    ext_3 = st.slider("External Credit Score 3 (0.0 - 1.0)", 0.0, 1.0, 0.58, 0.01)

# Calculate domain ratios matching notebook pipeline
days_employed = -emp_years * 365
credit_term = annuity / credit_amt if credit_amt > 0 else 0
credit_income_percent = credit_amt / income if income > 0 else 0

input_df = pd.DataFrame([{
    'EXT_SOURCE_1': ext_1,
    'EXT_SOURCE_2': ext_2,
    'EXT_SOURCE_3': ext_3,
    'AMT_INCOME_TOTAL': income,
    'AMT_CREDIT': credit_amt,
    'AMT_ANNUITY': annuity,
    'AGE_YEARS': age,
    'DAYS_EMPLOYED': days_employed,
    'CREDIT_TERM': credit_term,
    'CREDIT_INCOME_PERCENT': credit_income_percent
}])

st.divider()

if st.button("Predict Loan Eligibility", type="primary", use_container_width=True):
    default_prob = model.predict_proba(input_df)[0][1]
    risk_pct = default_prob * 100

    st.subheader("Assessment Result")
    if default_prob < 0.50:
        st.success(f"✅ **ELIGIBLE FOR LOAN** — Estimated Default Risk: **{risk_pct:.2f}%**")
        st.info("Applicant demonstrates strong external bureau scores and sustainable repayment ratios.")
    else:
        st.error(f"⚠️ **HIGH DEFAULT RISK (REJECT)** — Estimated Default Risk: **{risk_pct:.2f}%**")
        st.warning("Elevated default probability detected based on credit-to-income burden or bureau score profile.")