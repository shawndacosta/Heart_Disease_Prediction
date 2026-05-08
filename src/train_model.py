import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

train = pd.read_csv("train.csv")
train["Heart Disease"] = train["Heart Disease"].map({"Absence": 0, "Presence": 1})

X = train.drop(columns=["Heart Disease", "id"])
y = train["Heart Disease"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)

model = XGBClassifier(
    max_depth=3,
    n_estimators=200,
    learning_rate=0.2,
    random_state=42,
    base_score=0.5
)

model.fit(X_train, y_train)

preds = model.predict(X_valid)
print(f"Accuracy : {accuracy_score(y_valid, preds):.4f}")
print(f"F1 Score : {f1_score(y_valid, preds):.4f}")


joblib.dump(model, "heart_model.pkl")
print("Modèle sauvegardé → heart_model.pkl ✅")