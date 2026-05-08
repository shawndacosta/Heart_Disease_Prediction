import streamlit as st
import requests
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

FEATURE_NAMES = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
    "FBS over 120", "EKG results", "Max HR", "Exercise angina",
    "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium"
]

st.title("❤️ Heart Disease Risk Predictor")
st.markdown("Fill in the patient's clinical data to assess the risk of heart disease.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=58)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4])
    bp = st.number_input("Blood Pressure (BP)", min_value=50, max_value=250, value=130)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=250)
    fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ekg = st.selectbox("EKG Results", options=[0, 1, 2])

with col2:
    max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope_st = st.selectbox("Slope of ST", options=[1, 2, 3])
    vessels = st.selectbox("Number of Vessels (Fluoro)", options=[0, 1, 2, 3])
    thallium = st.selectbox("Thallium Test", options=[3, 6, 7])

st.divider()

if st.button("🔍 Analyze Risk", use_container_width=True):
    payload = {
        "Age": age,
        "Sex": sex,
        "Chest_pain_type": chest_pain,
        "BP": bp,
        "Cholesterol": cholesterol,
        "FBS_over_120": fbs,
        "EKG_results": ekg,
        "Max_HR": max_hr,
        "Exercise_angina": exercise_angina,
        "ST_depression": st_depression,
        "Slope_of_ST": slope_st,
        "Number_of_vessels_fluro": vessels,
        "Thallium": thallium
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        result = response.json()

        st.divider()
        risk = result["risk_score"]

        if result["prediction"] == 1:
            st.error("⚠️ High risk of heart disease detected")
        else:
            st.success("✅ No heart disease detected")

        st.metric(label="Risk Score", value=f"{risk:.1%}")
        st.progress(risk)

        st.divider()
        st.subheader("🔎 Explanation — Why this prediction?")

        import numpy as np
        shap_values = np.array(result["shap_values"])
        feature_values = np.array(result["feature_values"])
        base_value = result["base_value"]

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=feature_values,
                feature_names=FEATURE_NAMES
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"API connection error: {e}")
