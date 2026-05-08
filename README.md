# ❤️ Heart Disease Prediction – End-to-End AI Application

## Project Overview

This project focuses on predicting the presence of **heart disease** using structured **clinical data**.

Multiple **classification models** were implemented and compared through a rigorous evaluation pipeline including **cross-validation** and **hyperparameter tuning**. The model is then deployed as a **REST API** with an **interactive interface** featuring **SHAP explanations**, fully containerized with **Docker**.

## 📑 Table of Contents
1. [I. Introduction 🩺](#i-introduction-)
2. [II. Dataset 🔍](#ii-dataset-)
3. [III. Data Preprocessing 🛠️](#iii-data-preprocessing-%EF%B8%8F)
4. [IV. Model Selection & Hyperparameter Optimization 📈](#iv-model-selection--hyperparameter-optimization-)
5. [V. Models Evaluation 📊](#v-models-evaluation-)
6. [VI. API 🔌](#vi-api-)
7. [VII. Interface 🖥️](#vii-interface-%EF%B8%8F)
8. [VIII. Docker 🐳](#viii-docker-)
9. [IX. Conclusion ✔️](#ix-conclusion-%EF%B8%8F)

# I. Introduction 🩺

Cardiovascular diseases are among the leading causes of mortality worldwide. Early detection based on measurable clinical indicators can significantly improve patient outcomes.

The goal of this project is to build a **supervised machine learning model** capable of predicting **heart disease presence** using demographic and medical features, and to deploy it as a complete AI application.

# II. Dataset 🔍

- **Training set** : 630,000 rows, 13 clinical features + 1 target (`Heart Disease`)
- **Target distribution** : Absence 347,546 — Presence 282,454 (balanced)
- **No missing values** detected

Since the dataset is balanced, ***accuracy*** was used as the primary evaluation metric alongside ***precision***, ***recall***, and ***F1-score***.

# III. Data Preprocessing 🛠️

- **Encoding** : `Absence` → 0, `Presence` → 1
- **No data leakage** : only the target variable was encoded
- **Train/Validation split** : 80/20 with `random_state=42`

# IV. Model Selection & Hyperparameter Optimization 📈

Multiple classification models were evaluated with a predefined hyperparameter grid. Hyperparameter tuning was performed using **GridSearchCV** with a **5-fold cross-validation** strategy.

# V. Models Evaluation 📊

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **XGBoost ✅** | 0.887 | 0.880 | 0.866 | 0.873 |
| RandomForest | 0.884 | 0.879 | 0.861 | 0.870 |
| DecisionTree | 0.881 | 0.875 | 0.856 | 0.866 |

**XGBoost** was selected as the final model with the following hyperparameters:

| Parameter | Value |
|---|---|
| max_depth | 3 |
| n_estimators | 200 |
| learning_rate | 0.2 |

# VI. API 🔌

The trained model is exposed via a **REST API** built with **FastAPI**.

| Method | Route | Description |
|---|---|---|
| GET | `/` | Check if the API is running |
| POST | `/predict` | Predict heart disease risk from patient data |

### Example Request

```json
{
  "Age": 58,
  "Sex": 1,
  "Chest_pain_type": 4,
  "BP": 152,
  "Cholesterol": 239,
  "FBS_over_120": 0,
  "EKG_results": 0,
  "Max_HR": 158,
  "Exercise_angina": 1,
  "ST_depression": 3.6,
  "Slope_of_ST": 2,
  "Number_of_vessels_fluro": 2,
  "Thallium": 7
}
```

### Example Response

```json
{
  "risk_score": 0.997,
  "prediction": 1,
  "shap_values": [...],
  "base_value": 0.447,
  "feature_values": [58, 1, 4, 152, 239, 0, 0, 158, 1, 3.6, 2, 2, 7]
}
```

# VII. Interface 🖥️

An interactive **Streamlit** interface allows users to fill in patient clinical data and receive:

- A **risk score** (0 to 100%)
- A **prediction** (Heart Disease / No Heart Disease)
- A **SHAP waterfall plot** explaining which features contributed most to the prediction

# VIII. Docker 🐳

The full application is containerized with **Docker** and orchestrated with **docker-compose**.

### With Docker
```bash
git clone https://github.com/shawndacosta/Heart_Disease_Prediction
cd Heart_Disease_Prediction
docker-compose up --build
```
Open `http://localhost:8501`

### Without Docker
```bash
git clone https://github.com/shawndacosta/Heart_Disease_Prediction
cd Heart_Disease_Prediction
pip install -r requirements.txt
```

Then split a terminal:

```bash
# Terminal 1
uvicorn api:app --reload
```

```bash
# Terminal 2
streamlit run app.py
```

Open `http://localhost:8501`

# IX. Conclusion ✔️

This project demonstrates a complete **end-to-end AI workflow**, from data exploration to production deployment. After comparing multiple algorithms, **XGBoost** achieved the strongest overall performance. The model is served via a **FastAPI REST API**, accessible through a **Streamlit interface** with **SHAP explanations** per patient, and fully containerized with **Docker**.
