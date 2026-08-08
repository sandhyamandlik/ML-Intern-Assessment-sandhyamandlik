"""
evaluate.py
-----------
Generates the evaluation artifacts required by the assessment:
  - Confusion matrix
  - ROC curve
  - Feature importance
  - A written model comparison summary (reports/results_summary.md)

Run this AFTER train.py (it re-trains all four candidates in memory so it
can plot all of them; train.py already persisted the winning model).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

from preprocessing import get_train_test_split
from train import (
    tune_logistic_regression, tune_random_forest, tune_xgboost, evaluate,
)

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "reports" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    X_train, X_test, y_train, y_test = get_train_test_split()

    print("Re-fitting all 3 tuned models for full evaluation plots...")
    searches = {
        "logistic_regression": tune_logistic_regression(X_train, y_train),
        "random_forest": tune_random_forest(X_train, y_train),
        "xgboost": tune_xgboost(X_train, y_train),
    }
    fitted = {name: s.best_estimator_ for name, s in searches.items()}
    results = {name: evaluate(pipe, X_test, y_test) for name, pipe in fitted.items()}

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"Best model: {best_name}")

    # ---- Confusion matrices (grid of 3) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, pipe) in zip(axes, fitted.items()):
        preds = pipe.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "06_confusion_matrices.png", dpi=120)
    plt.close()

    # ---- ROC curves (all 3 overlaid) ----
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "07_roc_curves.png", dpi=120)
    plt.close()

    # ---- Feature importance (best model, if tree-based; else coefficients) ----
    best_pipe = fitted[best_name]
    preprocessor = best_pipe.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    model = best_pipe.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = None

    if importances is not None:
        order = np.argsort(importances)[-15:]  # top 15
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(order)), importances[order], color="#4C72B0")
        plt.yticks(range(len(order)), [feature_names[i] for i in order])
        plt.title(f"Top 15 Feature Importances — {best_name}")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "08_feature_importance.png", dpi=120)
        plt.close()

    # ---- Written summary ----
    lines = ["# Model Evaluation & Comparison\n"]
    lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    lines.append("|---|---|---|---|---|---|")
    for name, m in results.items():
        marker = " **(selected)**" if name == best_name else ""
        lines.append(
            f"| {name}{marker} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} |"
        )
    lines.append(f"\n## Final Model Selection: `{best_name}`\n")
    lines.append(
        "Selected on **ROC-AUC** on the held-out test set. Rationale: the target class is "
        "imbalanced (~73% retained / ~27% churned), so accuracy alone is a misleading metric "
        "(a model that always predicts \"No Churn\" would still score ~73% accuracy). ROC-AUC "
        "measures ranking quality across all classification thresholds and is threshold-"
        "independent, which matters here because a business would likely tune the decision "
        "threshold based on the cost of a retention offer vs. the cost of losing a customer, "
        "rather than accepting the default 0.5 cutoff.\n"
    )
    lines.append(
        "Recall is also reported prominently: in a churn-prevention business context, missing "
        "an actual churner (false negative) is typically more costly than flagging a loyal "
        "customer for a retention offer (false positive), so we favor a model with strong "
        "recall on the positive (churn) class over one that is simply \"more accurate\" overall.\n"
    )

    with open(ROOT / "reports" / "results_summary.md", "w") as f:
        f.write("\n".join(lines))

    with open(ROOT / "reports" / "model_comparison.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2, default=str)

    print("Evaluation complete. Plots + results_summary.md written to reports/.")


if __name__ == "__main__":
    main()
