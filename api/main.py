from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load both models at startup
artifacts = {
    "logistic": joblib.load("models/logistic_regression_model.pkl"),
    "xgboost": joblib.load("models/xgboost_model.pkl"),
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Student risk prediction API", "models": list(artifacts.keys())}

class StudentData(BaseModel):
    Student_ID: int
    Age: int
    Gender: str
    Department: str
    CGPA: float
    Sleep_Duration: float
    Study_Hours: float
    Social_Media_Hours: float
    Physical_Activity: float
    Stress_Level: float

@app.post("/predict")
def predict(
    student_data: StudentData,
    model: str = Query(default="logistic", enum=["logistic", "xgboost"])
):
    artifact = artifacts[model]
    m = artifact["model"]
    scaler = artifact["scaler"]
    threshold = artifact["threshold"]
    columns = artifact["columns"]

    input_data = pd.DataFrame([student_data.dict()])
    categorical_columns = input_data.select_dtypes(include=['object']).columns.tolist()
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns, dtype=float).reindex(columns=columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data_encoded)

    y_pred_proba = m.predict_proba(input_data_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    risk_status = "At Risk" if y_pred[0] == 1 else "Not At Risk"
    return {"model": model, "prediction": risk_status, "probability": float(y_pred_proba[0])}