# Exploratory Data Analysis Summary

## Dataset Overview

- Shape: 7043 rows x 21 columns

- Target column: `Churn` (Yes/No)


## Data Types

|                  | dtype   |
|:-----------------|:--------|
| customerID       | str     |
| gender           | str     |
| SeniorCitizen    | int64   |
| Partner          | str     |
| Dependents       | str     |
| tenure           | int64   |
| PhoneService     | str     |
| MultipleLines    | str     |
| InternetService  | str     |
| OnlineSecurity   | str     |
| OnlineBackup     | str     |
| DeviceProtection | str     |
| TechSupport      | str     |
| StreamingTV      | str     |
| StreamingMovies  | str     |
| Contract         | str     |
| PaperlessBilling | str     |
| PaymentMethod    | str     |
| MonthlyCharges   | float64 |
| TotalCharges     | str     |
| Churn            | str     |


## Data Quality Issues Found

- `TotalCharges` is stored as a string and contains 11 blank entries (all customers with `tenure == 0`, i.e. brand-new customers who have not been billed yet). This is not "missing at random" -- it is structurally explainable, so we impute it as 0 rather than dropping rows.

- Duplicate records (excluding customerID): 22

- Native missing values (NaN) elsewhere: {'TotalCharges_numeric': 11}


## Target Class Balance

- No: 73.5%  |  Yes: 26.5%

- The dataset is moderately imbalanced (~73/27). We account for this with `class_weight='balanced'` / `scale_pos_weight` in modeling and by reporting F1 and ROC-AUC rather than relying on accuracy alone.


## Outlier Detection

- Boxplots on `tenure`, `MonthlyCharges`, `TotalCharges` show no extreme outliers requiring removal -- values are naturally bounded (tenure in months, charges in a realistic USD range). We therefore do not drop rows, but we do apply `StandardScaler` since some models (Logistic Regression, SVM-family) are scale-sensitive.


## Correlation Analysis

- `tenure` has the strongest (negative) correlation with churn -- customers who have stayed longer are far less likely to leave.

- `MonthlyCharges` is mildly positively correlated with churn.

- `TotalCharges` is highly correlated with `tenure` (tenure x monthly spend), which is worth flagging for potential multicollinearity in linear models.


## Key Observations from Categorical Breakdown

- Month-to-month contracts churn far more than one/two-year contracts -- contract length is one of the strongest behavioral predictors.

- Fiber optic internet customers churn more than DSL customers, despite (or because of) higher pricing -- worth flagging as a business insight, not just a modeling feature.

- Electronic check payers churn noticeably more than customers on automatic payment methods (bank transfer / credit card).
