"""
predict.py
----------
Reusable prediction pipeline. Loads the saved model (preprocessor + model
bundled together as one sklearn Pipeline, from train.py) and exposes a
single `predict_churn()` function that any caller (CLI, Streamlit app,
API, tests) can use with a plain dict of raw customer fields -- no manual
re-implementation of the encoding/scaling logic anywhere else.
"""

from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "saved_model" / "churn_model.joblib"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_churn(customer: dict) -> dict:
    """
    customer: dict with the 19 raw feature keys used in training
              (see preprocessing.NUMERIC_FEATURES + CATEGORICAL_FEATURES).
    Returns: {"churn_prediction": "Yes"/"No", "churn_probability": float}
    """
    model = _get_model()
    df = pd.DataFrame([customer])
    proba = model.predict_proba(df)[0, 1]
    pred = "Yes" if proba >= 0.5 else "No"
    return {"churn_prediction": pred, "churn_probability": round(float(proba), 4)}


if __name__ == "__main__":
    # Smoke test with a sample customer likely to churn
    sample = {
        "tenure": 2,
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    print(predict_churn(sample))
