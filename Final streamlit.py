# insurance_multi_dashboard.py
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import string, joblib, nltk
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder, MinMaxScaler, StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# One global page‑config (must be first Streamlit command)
st.set_page_config(page_title="Insurance Analytics Suite", layout="centered")

# ─────────────────────────────────────────────
# PATHS (Fraud model)
# ─────────────────────────────────────────────
DATA_FRAUD  = Path("insurance_fraud_featured_dataset.csv")
MODEL_FRAUD = Path("decision_tree_pipeline.pkl")

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def bucket_age(age: int) -> str:
    return ("Under25" if age < 25 else
            "25-40"  if age < 40 else
            "40-60"  if age < 60 else "60+")

# ─────────────────────────────────────────────
# 1)  FRAUD PREDICTOR
# ─────────────────────────────────────────────
def fraud_tool():
    st.header("🚨 Insurance Fraud Predictor")

    # Load or train model
    if MODEL_FRAUD.exists():
        model = joblib.load(MODEL_FRAUD)
    else:
        st.info("Training fraud model (first run)…")
        df = pd.read_csv(DATA_FRAUD)
        num_cols = ["Annual_Income", "Customer_Age", "Claim_Amount",
                    "Premium_Amount", "Claim_History"]
        cat_cols = ["Policy_Type", "Gender", "Age_Group"]
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df[cat_cols] = df[cat_cols].apply(lambda s: s.fillna(s.mode()[0]))
        df["Risk_Score"] = df["Risk_Score"].map({"Low": 0, "Medium": 1, "High": 2})
        X, y = df.drop(columns=["Fraudulent_Claim"]), df["Fraudulent_Claim"]

        num_pipe = Pipeline([("imp", MinMaxScaler())])
        cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
        preproc = ColumnTransformer([("num", num_pipe, num_cols),
                                     ("cat", cat_pipe, cat_cols)],
                                    remainder="passthrough")
        model = Pipeline([("prep", preproc),
                          ("clf",  DecisionTreeClassifier(
                              max_depth=6, class_weight="balanced", random_state=42))])
        model.fit(X, y)
        joblib.dump(model, MODEL_FRAUD)
        st.success("Fraud model trained & saved")

    # Input UI
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 100, 35)
        inc = st.number_input("Annual Income (₹)", 50_000, 5_000_000, step=50_000)
        claim = st.number_input("Claim Amount (₹)", 1_000, 1_000_000, step=1_000)
        prem = st.number_input("Premium Amount (₹)", 500, 200_000, step=500)
    with col2:
        hist = st.slider("Past Claims", 0, 10, 0)
        ptype = st.selectbox("Policy Type", ["Auto", "Home", "Life", "Travel", "Health"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        risk_txt = st.selectbox("Risk Score (UW)", ["Low", "Medium", "High"])

    if st.button("🔍 Predict Fraud"):
        df = pd.DataFrame([{
            "Customer_Age": age, "Annual_Income": inc, "Claim_Amount": claim,
            "Premium_Amount": prem, "Claim_History": hist, "Policy_Type": ptype,
            "Gender": gender, "Risk_Score": risk_txt
        }])
        df["Claim_to_Premium_Ratio"] = df["Claim_Amount"] / df["Premium_Amount"]
        df["Income_to_Claim_Ratio"]  = df["Annual_Income"] / df["Claim_Amount"]
        df["Has_Prior_Claims"]       = (df["Claim_History"] > 0).astype(int)
        df["Age_Group"] = bucket_age(age)
        df["Risk_Score"] = df["Risk_Score"].map({"Low": 0, "Medium": 1, "High": 2})

        order = ["Customer_Age", "Annual_Income", "Claim_Amount", "Premium_Amount",
                 "Claim_History", "Policy_Type", "Gender", "Age_Group",
                 "Risk_Score", "Claim_to_Premium_Ratio",
                 "Income_to_Claim_Ratio", "Has_Prior_Claims"]
        pred  = model.predict(df[order])[0]
        proba = model.predict_proba(df[order])[0][1]
        if pred == 1:
            st.error(f"⚠️ Fraud detected! Confidence: **{proba:.1%}**")
        else:
            st.success(f"✅ Claim Genuine. Confidence: **{(1-proba):.1%}**")

# ─────────────────────────────────────────────
# 2)  SENTIMENT ANALYZER
# ─────────────────────────────────────────────
def sentiment_tool():
    st.header("🗣️ Review Sentiment Analyzer")

    # NLP resources
    nltk.download("punkt"); nltk.download("stopwords"); nltk.download("wordnet", quiet=True)
    stop_words = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    def clean(t):
        t = str(t).lower().translate(str.maketrans("", "", string.punctuation))
        tok = [lem.lemmatize(w) for w in nltk.word_tokenize(t)
               if w.isalpha() and w not in stop_words]
        return " ".join(tok)

    file = st.file_uploader("Upload CSV with Review_Text & Rating", type="csv")
    if not file: return
    df = pd.read_csv(file)
    if {"Review_Text", "Rating"} - set(df.columns):
        st.error("CSV needs Review_Text and Rating columns"); return

    with st.spinner("Training sentiment model…"):
        df["Clean"] = df["Review_Text"].apply(clean)
        df["Label"] = df["Rating"].apply(lambda r: 0 if r <= 2 else 1 if r == 3 else 2)
        tfidf = TfidfVectorizer(max_features=3000)
        X = tfidf.fit_transform(df["Clean"]); y = df["Label"]
        Xb, yb = SMOTE(random_state=42).fit_resample(X, y)
        Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.2, random_state=42)
        mdl = RandomForestClassifier(n_estimators=150, random_state=42).fit(Xtr, ytr)
    st.success("Model trained!")

    review = st.text_area("Write a review")
    if st.button("Analyze"):
        pred = mdl.predict(tfidf.transform([clean(review)]))[0]
        st.info(f"Sentiment: **{ {0:'Negative',1:'Neutral',2:'Positive'}[pred] }**")

# ─────────────────────────────────────────────
# 3)  POLICY‑UPGRADES REGRESSOR
# ─────────────────────────────────────────────
def upgrades_tool():
    st.header("📈 Policy‑Upgrades Predictor")

    file = st.file_uploader("Upload CSV dataset", type="csv")
    if not file: return
    df = pd.read_csv(file)
    if "Policy_Upgrades" not in df.columns:
        st.error("'Policy_Upgrades' column missing."); return

    id_like = [c for c in df.columns if "id" in c.lower()]
    df.drop(columns=id_like, inplace=True, errors="ignore")
    features = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != "Policy_Upgrades"]
    if not features:
        st.error("No numeric feature columns."); return

    X, y = df[features], df["Policy_Upgrades"]
    scaler = StandardScaler().fit(X)
    mdl = RandomForestRegressor(random_state=42).fit(scaler.transform(X), y)
    st.success("Model trained! Enter values:")

    user_vals = {c: st.number_input(c, float(df[c].min()),
                                    float(df[c].max()), float(df[c].mean()))
                 for c in features}
    if st.button("Predict Upgrades"):
        pred = int(round(mdl.predict(scaler.transform(pd.DataFrame([user_vals])))[0]))
        st.success(f"Predicted Upgrades: **{pred}**")

# ─────────────────────────────────────────────
# Sidebar router
# ─────────────────────────────────────────────
tool = st.sidebar.radio(
    "Choose a tool:",
    ("Fraud Predictor", "Sentiment Analyzer", "Policy Upgrades")
)

if tool == "Fraud Predictor":
    fraud_tool()
elif tool == "Sentiment Analyzer":
    sentiment_tool()
else:
    upgrades_tool()
