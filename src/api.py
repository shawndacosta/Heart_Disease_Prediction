from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
from xgboost import XGBClassifier

app = FastAPI(title="Heart Disease Prediction API")

model = joblib.load("heart_model.pkl")
explainer = shap.TreeExplainer(model)

FEATURE_NAMES = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
    "FBS over 120", "EKG results", "Max HR", "Exercise angina",
    "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium"
]

class PatientData(BaseModel):
    Age: int
    Sex: int
    Chest_pain_type: int
    BP: int
    Cholesterol: int
    FBS_over_120: int
    EKG_results: int
    Max_HR: int
    Exercise_angina: int
    ST_depression: float
    Slope_of_ST: int
    Number_of_vessels_fluro: int
    Thallium: int

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running ✅"}

@app.post("/predict")
def predict(patient: PatientData):
    data = pd.DataFrame([{
        "Age": patient.Age,
        "Sex": patient.Sex,
        "Chest pain type": patient.Chest_pain_type,
        "BP": patient.BP,
        "Cholesterol": patient.Cholesterol,
        "FBS over 120": patient.FBS_over_120,
        "EKG results": patient.EKG_results,
        "Max HR": patient.Max_HR,
        "Exercise angina": patient.Exercise_angina,
        "ST depression": patient.ST_depression,
        "Slope of ST": patient.Slope_of_ST,
        "Number of vessels fluro": patient.Number_of_vessels_fluro,
        "Thallium": patient.Thallium
    }])

    proba = model.predict_proba(data)[0][1]
    prediction = int(proba >= 0.5)

    shap_values = explainer.shap_values(data)
    shap_list = shap_values[0].tolist()

    return {
        "risk_score": round(float(proba), 3),
        "prediction": prediction,
        "result": "Heart Disease Detected ⚠️" if prediction == 1 else "No Heart Disease ✅",
        "shap_values": shap_list,
        "base_value": float(explainer.expected_value),
        "feature_values": data.iloc[0].tolist()
    }