# Home Loan Default Risk Prediction & Eligibility Analysis (PRCP-1006)

> **Live Web App:** [Launch Streamlit App](#) *(Deployment link coming soon)*

An end-to-end banking Machine Learning system designed to evaluate borrower creditworthiness, identify key risk factors, and predict loan default risk across **307,511 loan applications** and **7 interconnected relational datasets** (`application_train`, `bureau`, `bureau_balance`, `previous_application`, `installments_payments`, `POS_CASH_balance`, and `credit_card_balance`).

## Problem Context & Dataset
Financial institutions face high default risk when evaluating applicants with limited credit histories. Using 7 relational banking tables linked by `SK_ID_CURR` and `SK_ID_BUREAU`, this project uncovers behavioral repayment patterns and builds an automated default risk scoring pipeline (`TARGET = 1` for defaulter, `TARGET = 0` for non-defaulter).

## Technical Approach
* **Data Anomaly & Sparsity Resolution:** Identified and resolved the `DAYS_EMPLOYED = 365243` anomaly (18.01% of rows) via binary flagging (`DAYS_EMPLOYED_ANOM`) and `NaN` imputation, and removed columns exceeding 65% missingness.
* **Multi-Table Relational Aggregation:** Hierarchically aggregated historical credit bureau, installment payment (`DPD`/`DBD`), POS/cash, and credit card utilization tables to the applicant level (`SK_ID_CURR`).
* **Domain Feature Engineering:** Created financial ratios including `CREDIT_TERM`, `CREDIT_INCOME_PERCENT`, `ANNUITY_INCOME_PERCENT`, `DAYS_EMPLOYED_PERCENT`, and `EXT_SOURCES_MEAN`.
* **Imbalance Handling:** Addressed severe class imbalance (91.93% non-defaulters vs. 8.07% defaulters) using stratified train-validation splitting and `class_weight='balanced'`.

## Model Comparison Results

| Model | ROC-AUC | Recall (Defaulter) | Precision (Defaulter) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 0.7646 | 0.6931 | 0.1689 | 0.2716 |
| **Random Forest (Balanced)** | 0.7568 | 0.6085 | 0.1875 | 0.2866 |
| **LightGBM (Balanced - Production)** | **0.7854** | 0.6518 | **0.2005** | **0.3067** |

## Key Eligibility Drivers
* **Loan Repayment Structure:** `CREDIT_TERM` (`AMT_ANNUITY / AMT_CREDIT`) is the #1 predictive feature driving default risk.
* **External Bureau Scores:** `EXT_SOURCES_MEAN` and individual normalized external scores (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`) strongly separate prime borrowers from high-risk segments.
* **Historical Payment & Tenure Stability:** Minimum historical installment payments (`INSTAL_AMT_PAYMENT_MIN`) and employment tenure ratio (`DAYS_EMPLOYED_PERCENT`) reliably distinguish consistent borrowers.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
