"""
train.py
--------
Trains a baseline model plus three candidate models, tunes each with
GridSearchCV (5-fold stratified CV), compares them, and saves the best
pipeline (preprocessor + model bundled together) to models/saved_model/.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier

from preprocessing import get_train_test_split, build_preprocessor

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "saved_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_pipeline(estimator):
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])


def run_baseline(X_train, y_train, X_test, y_test):
    """
    Baseline: plain Logistic Regression, no tuning. Establishes the floor
    that the tuned models must beat to justify their extra complexity.
    """
    pipe = make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    return metrics


def tune_logistic_regression(X_train, y_train):
    pipe = make_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"))
    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs"],
    }
    search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=CV, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def tune_random_forest(X_train, y_train):
    pipe = make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"))
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [6, 10, None],
        "model__min_samples_leaf": [1, 3],
    }
    search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=CV, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def tune_xgboost(X_train, y_train):
    # scale_pos_weight ~ (negative / positive) count ratio, to address the
    # class imbalance noted in EDA (~73/27).
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    pipe = make_pipeline(
        XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
    )
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
    }
    search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=CV, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def evaluate(pipe, X_test, y_test):
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    X_train, X_test, y_train, y_test = get_train_test_split()

    results = {}
    fitted = {}

    print("Training baseline (Logistic Regression, untuned)...")
    results["baseline_logreg"] = run_baseline(X_train, y_train, X_test, y_test)

    print("Tuning Logistic Regression...")
    t0 = time.time()
    lr_search = tune_logistic_regression(X_train, y_train)
    fitted["logistic_regression"] = lr_search.best_estimator_
    results["logistic_regression"] = evaluate(lr_search.best_estimator_, X_test, y_test)
    results["logistic_regression"]["cv_best_roc_auc"] = lr_search.best_score_
    results["logistic_regression"]["best_params"] = lr_search.best_params_
    print(f"  done in {time.time()-t0:.1f}s, best params: {lr_search.best_params_}")

    print("Tuning Random Forest...")
    t0 = time.time()
    rf_search = tune_random_forest(X_train, y_train)
    fitted["random_forest"] = rf_search.best_estimator_
    results["random_forest"] = evaluate(rf_search.best_estimator_, X_test, y_test)
    results["random_forest"]["cv_best_roc_auc"] = rf_search.best_score_
    results["random_forest"]["best_params"] = rf_search.best_params_
    print(f"  done in {time.time()-t0:.1f}s, best params: {rf_search.best_params_}")

    print("Tuning XGBoost...")
    t0 = time.time()
    xgb_search = tune_xgboost(X_train, y_train)
    fitted["xgboost"] = xgb_search.best_estimator_
    results["xgboost"] = evaluate(xgb_search.best_estimator_, X_test, y_test)
    results["xgboost"]["cv_best_roc_auc"] = xgb_search.best_score_
    results["xgboost"]["best_params"] = xgb_search.best_params_
    print(f"  done in {time.time()-t0:.1f}s, best params: {xgb_search.best_params_}")

    # ---- Select final model ----
    # Selection metric: ROC-AUC on the held-out test set. ROC-AUC is chosen
    # over raw accuracy because the classes are imbalanced (~73/27) and the
    # business cost of missing a churner (false negative) is higher than a
    # false positive (an unnecessary retention offer) -- ROC-AUC captures
    # ranking quality across all thresholds rather than one fixed cutoff.
    candidates = {k: v for k, v in results.items() if k != "baseline_logreg"}
    best_name = max(candidates, key=lambda k: candidates[k]["roc_auc"])
    best_pipe = fitted[best_name]

    print(f"\nBest model selected: {best_name}")
    print(json.dumps(results[best_name], indent=2, default=str))

    joblib.dump(best_pipe, MODEL_DIR / "churn_model.joblib")
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "numeric_features": ["tenure", "MonthlyCharges", "TotalCharges"],
                "categorical_features": [
                    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod",
                ],
            },
            f,
            indent=2,
        )

    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved best model ({best_name}) to {MODEL_DIR / 'churn_model.joblib'}")
    return results, best_name, X_test, y_test, fitted


if __name__ == "__main__":
    main()
