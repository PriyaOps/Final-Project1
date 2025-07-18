import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Policy Upgrades Predictor", layout="centered")
st.title("🔧 Customer Segmentation")

# 📤 Upload CSV
uploaded_file = st.file_uploader("📥 Upload your dataset (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Drop ID columns if any
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df.drop(columns=id_cols, inplace=True, errors='ignore')

    # Ensure 'Policy_Upgrades' exists
    if 'Policy_Upgrades' not in df.columns:
        st.error("❌ 'Policy_Upgrades' column not found in the dataset.")
        st.stop()

    # Select target and feature columns
    target_col = 'Policy_Upgrades'
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]

    if len(feature_cols) == 0:
        st.error("❌ No numerical feature columns found besides 'Policy_Upgrades'.")
        st.stop()

    X = df[feature_cols]
    y = df[target_col]

    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_scaled, y)

    st.subheader("📥 Enter customer details for prediction")

    # Input fields for features
    user_input = {}
    for col in feature_cols:
        col_data = df[col]
        min_val, max_val, mean_val = float(col_data.min()), float(col_data.max()), float(col_data.mean())
        user_input[col] = st.number_input(f"{col}", min_val, max_val, mean_val)

    # Prediction
    if st.button("🚀 Predict Policy Upgrades"):
        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)[0]
        rounded_prediction = int(round(prediction))
        st.success(f"🎯 Predicted Policy Upgrades: **{rounded_prediction}**")





