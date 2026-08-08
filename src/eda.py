"""
eda.py
------
Exploratory Data Analysis for the Telco Customer Churn dataset.

Why this dataset:
    Telco Customer Churn is a classic, well-understood binary classification
    problem (7,043 records, 21 columns) with a realistic mix of categorical,
    numerical, and "dirty" data (TotalCharges is stored as text and contains
    blank strings for brand-new customers) -- which forces genuine cleaning
    decisions rather than a dataset that "just works" out of the box.

This script produces:
    - reports/images/*.png  (distribution, correlation, churn-rate plots)
    - reports/eda_summary.md (written observations)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Telco-Customer-Churn.csv"
IMG_DIR = Path(__file__).resolve().parent.parent / "reports" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_raw():
    return pd.read_csv(DATA_PATH)


def run_eda():
    df = load_raw()
    lines = []

    lines.append(f"## Dataset Overview\n")
    lines.append(f"- Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    lines.append(f"- Target column: `Churn` (Yes/No)\n")

    # dtypes
    lines.append("\n## Data Types\n")
    lines.append(df.dtypes.to_frame("dtype").to_markdown())

    # TotalCharges is object type but should be numeric -- 11 rows have blank strings
    df["TotalCharges_numeric"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_bad_totalcharges = df["TotalCharges_numeric"].isna().sum()
    lines.append(f"\n\n## Data Quality Issues Found\n")
    lines.append(f"- `TotalCharges` is stored as a string and contains "
                  f"{n_bad_totalcharges} blank entries (all customers with `tenure == 0`, "
                  f"i.e. brand-new customers who have not been billed yet). "
                  f"This is not \"missing at random\" -- it is structurally explainable, "
                  f"so we impute it as 0 rather than dropping rows.\n")

    # duplicates
    n_dupes = df.duplicated(subset=[c for c in df.columns if c != "customerID"]).sum()
    lines.append(f"- Duplicate records (excluding customerID): {n_dupes}\n")

    # missing values overall
    missing = df.isna().sum()
    missing = missing[missing > 0]
    lines.append(f"- Native missing values (NaN) elsewhere: "
                  f"{'none' if missing.empty else missing.to_dict()}\n")

    # class balance
    churn_counts = df["Churn"].value_counts(normalize=True) * 100
    lines.append(f"\n## Target Class Balance\n")
    lines.append(f"- No: {churn_counts.get('No', 0):.1f}%  |  Yes: {churn_counts.get('Yes', 0):.1f}%\n")
    lines.append("- The dataset is moderately imbalanced (~73/27). We account for this with "
                  "`class_weight='balanced'` / `scale_pos_weight` in modeling and by reporting "
                  "F1 and ROC-AUC rather than relying on accuracy alone.\n")

    # ---- Plots ----
    # 1. Churn distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
    plt.title("Target Class Distribution (Churn)")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "01_churn_distribution.png", dpi=120)
    plt.close()

    # 2. Numeric feature distributions
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges_numeric"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, num_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "02_numeric_distributions.png", dpi=120)
    plt.close()

    # 3. Outlier check via boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, num_cols):
        sns.boxplot(y=df[col].dropna(), ax=ax, color="#DD8452")
        ax.set_title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "03_outlier_boxplots.png", dpi=120)
    plt.close()
    lines.append("\n## Outlier Detection\n")
    lines.append("- Boxplots on `tenure`, `MonthlyCharges`, `TotalCharges` show no extreme "
                  "outliers requiring removal -- values are naturally bounded (tenure in "
                  "months, charges in a realistic USD range). We therefore do not drop rows, "
                  "but we do apply `StandardScaler` since some models (Logistic Regression, "
                  "SVM-family) are scale-sensitive.\n")

    # 4. Correlation heatmap (numeric + binary-encoded target)
    corr_df = df.copy()
    corr_df["Churn_binary"] = (corr_df["Churn"] == "Yes").astype(int)
    corr_cols = ["tenure", "MonthlyCharges", "TotalCharges_numeric", "SeniorCitizen", "Churn_binary"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix (numeric features vs. Churn)")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "04_correlation_heatmap.png", dpi=120)
    plt.close()
    lines.append("\n## Correlation Analysis\n")
    lines.append("- `tenure` has the strongest (negative) correlation with churn -- customers "
                  "who have stayed longer are far less likely to leave.\n")
    lines.append("- `MonthlyCharges` is mildly positively correlated with churn.\n")
    lines.append("- `TotalCharges` is highly correlated with `tenure` (tenure x monthly spend), "
                  "which is worth flagging for potential multicollinearity in linear models.\n")

    # 5. Churn rate by key categorical features
    cat_features = ["Contract", "InternetService", "PaymentMethod"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, cat_features):
        rate = df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean() * 100).sort_values()
        rate.plot(kind="barh", ax=ax, color="#55A868")
        ax.set_title(f"Churn Rate (%) by {col}")
        ax.set_xlabel("Churn Rate (%)")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "05_churn_rate_by_category.png", dpi=120)
    plt.close()
    lines.append("\n## Key Observations from Categorical Breakdown\n")
    lines.append("- Month-to-month contracts churn far more than one/two-year contracts -- "
                  "contract length is one of the strongest behavioral predictors.\n")
    lines.append("- Fiber optic internet customers churn more than DSL customers, despite (or "
                  "because of) higher pricing -- worth flagging as a business insight, not just "
                  "a modeling feature.\n")
    lines.append("- Electronic check payers churn noticeably more than customers on automatic "
                  "payment methods (bank transfer / credit card).\n")

    with open(Path(__file__).resolve().parent.parent / "reports" / "eda_summary.md", "w") as f:
        f.write("# Exploratory Data Analysis Summary\n\n")
        f.write("\n".join(lines))

    print("EDA complete. Plots saved to reports/images/, summary saved to reports/eda_summary.md")


if __name__ == "__main__":
    run_eda()
