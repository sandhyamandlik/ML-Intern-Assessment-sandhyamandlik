# Model Evaluation & Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| logistic_regression | 0.739 | 0.504 | 0.782 | 0.613 | 0.840 |
| random_forest | 0.742 | 0.509 | 0.790 | 0.619 | 0.841 |
| xgboost **(selected)** | 0.758 | 0.529 | 0.796 | 0.635 | 0.843 |

## Final Model Selection: `xgboost`

Selected on **ROC-AUC** on the held-out test set. Rationale: the target class is imbalanced (~73% retained / ~27% churned), so accuracy alone is a misleading metric (a model that always predicts "No Churn" would still score ~73% accuracy). ROC-AUC measures ranking quality across all classification thresholds and is threshold-independent, which matters here because a business would likely tune the decision threshold based on the cost of a retention offer vs. the cost of losing a customer, rather than accepting the default 0.5 cutoff.

Recall is also reported prominently: in a churn-prevention business context, missing an actual churner (false negative) is typically more costly than flagging a loyal customer for a retention offer (false positive), so we favor a model with strong recall on the positive (churn) class over one that is simply "more accurate" overall.
