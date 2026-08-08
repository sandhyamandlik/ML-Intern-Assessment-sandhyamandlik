"""
preprocessing.py
-----------------
Cleaning, feature engineering, encoding, and scaling for the Telco
Customer Churn dataset. Returns a reusable sklearn ColumnTransformer
(the "preprocessor") that is saved and reused at inference time, so the
exact same transformations are applied to training data and to any new
customer record at prediction time.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Telco-Customer-Churn.csv"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
# SeniorCitizen is 0/1 already but we treat it as categorical (it's not an ordered magnitude)
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"


def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1) customerID is a pure identifier -- no predictive value, drop it.
    df = df.drop(columns=["customerID"])

    # 2) TotalCharges is stored as string; blanks correspond to tenure == 0
    #    (brand-new customers who have not been billed a full cycle yet).
    #    We coerce to numeric and impute those with 0, which is the true
    #    value for a customer with zero completed billing cycles.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 3) Deduplicate on all columns except the (already-dropped) ID.
    df = df.drop_duplicates()

    # 4) SeniorCitizen ships as 0/1 int -- cast to string so it is handled
    #    consistently by the categorical (OneHotEncoder) branch instead of
    #    being treated as a continuous numeric feature.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # 5) Target encoding: Yes/No -> 1/0
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """
    Why this approach:
      - StandardScaler on numeric features: Logistic Regression / SVM-style
        models are distance/gradient-scale sensitive; tree ensembles are
        unaffected by scaling but it does no harm to apply it uniformly
        through one shared preprocessing pipeline used by all models.
      - OneHotEncoder(handle_unknown='ignore') on categoricals: keeps the
        pipeline robust to any unseen category value at inference time
        instead of raising an error in production.
    """
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_train_test_split(test_size=0.2, random_state=42):
    df = load_and_clean()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    # stratify=y because of the ~73/27 class imbalance noted in EDA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_and_clean()
    print(df.shape)
    print(df.isna().sum().sum(), "missing values after cleaning")
    X_train, X_test, y_train, y_test = get_train_test_split()
    print("Train:", X_train.shape, "Test:", X_test.shape)
