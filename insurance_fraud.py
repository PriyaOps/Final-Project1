
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="🚨 Insurance Fraud Predictor", layout="centered")

st.title("🚨 Insurance Fraud Detection App")
st.markdown(
    "Enter claim details below. The model will classify the "
    "**Fraudulent** or **Genuine** ."
)

MODEL_PATH = Path("decision_tree_pipeline.pkl")
if not MODEL_PATH.exists():
    st.error("❌ Model file not found. Train the model first (see train_pipeline.py).")
    st.stop()

model = joblib.load(MODEL_PATH)

def build_features(raw: dict) -> pd.DataFrame:
    """Take raw user inputs → full feature table exactly like training."""
    df = pd.DataFrame([raw])

  
    df["Claim_to_Premium_Ratio"] = df["Claim_Amount"] / df["Premium_Amount"]
    df["Income_to_Claim_Ratio"]  = df["Annual_Income"] / df["Claim_Amount"]
    df["Has_Prior_Claims"]       = (df["Claim_History"] > 0).astype(int)

    
    age = df.loc[0, "Customer_Age"]
    if age < 25:
        df["Age_Group"] = "Under25"
    elif age < 40:
        df["Age_Group"] = "25-40"
    elif age < 60:
        df["Age_Group"] = "40-60"
    else:
        df["Age_Group"] = "60+"

    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    df["Risk_Score"] = df["Risk_Score"].map(risk_map)

  
    ordered_cols = [
        "Customer_Age", "Annual_Income", "Claim_Amount", "Premium_Amount",
        "Claim_History", "Policy_Type", "Gender", "Age_Group",
        "Risk_Score", "Claim_to_Premium_Ratio",
        "Income_to_Claim_Ratio", "Has_Prior_Claims"
    ]
    return df[ordered_cols]

st.header("📋 Enter Claim Details")

cust_age  = st.slider("Customer Age", 18, 100, 35)
income    = st.number_input("Annual Income (₹)", 50_000, 5_000_000, step=50_000)
claim_amt = st.number_input("Claim Amount (₹)",  1_000, 1_000_000, step=1_000)
prem_amt  = st.number_input("Premium Amount (₹)",   500,   200_000, step=500)
hist      = st.slider("Claim History (number of prior claims)", 0, 10, 0)

ptype  = st.selectbox("Policy Type", ["Auto", "Home", "Life", "Travel", "Health"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
risk   = st.selectbox("Risk Score (as rated by underwriting)", ["Low", "Medium", "High"])

if st.button("🔍 Predict"):
    raw_input = {
        "Customer_Age":   cust_age,
        "Annual_Income":  income,
        "Claim_Amount":   claim_amt,
        "Premium_Amount": prem_amt,
        "Claim_History":  hist,
        "Policy_Type":    ptype,
        "Gender":         gender,
        "Risk_Score":     risk,   # string; converted later
    }

    features = build_features(raw_input)
    pred  = model.predict(features)[0]           # 0 = genuine, 1 = fraud
    proba = model.predict_proba(features)[0][1]  

    if pred == 1:
        st.error(f"⚠️ **Fraudulent claim detected!")
    else:
        st.success(f"✅ Claim classified as **Genuine**.")


