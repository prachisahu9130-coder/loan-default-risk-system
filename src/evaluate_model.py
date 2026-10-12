
import numpy as np
import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "german.data"
MODEL_PATH = BASE_DIR / "models" / "loan_risk_model.pkl"

# Column names for the German Credit dataset
columns = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker",
    "target",
]

# Load the original dataset
df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=columns,
)

# Match the target mapping used to train the model:
# 1 = Good credit, 2 = Bad credit
df["target"] = df["target"].map({1: 0, 2: 1})

X = df.drop(columns=["target"])
y = df["target"]

# Recreate the same test split used during training.
# This assumes the model was trained using test_size=0.20
# and random_state=42. Change these if your training code differs.
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = joblib.load(MODEL_PATH)

y_probability = model.predict_proba(X_test)[:, 1]

# The project uses a 0.35 threshold for classification.
threshold = 0.35
y_pred = (y_probability >= threshold).astype(int)
# Save evaluation outputs for the deployed Streamlit app
EVALUATION_DIR = BASE_DIR / "models" / "evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

np.save(EVALUATION_DIR / "y_test.npy", y_test.to_numpy())
np.save(EVALUATION_DIR / "y_probability.npy", y_probability)

metrics = {
    "threshold": threshold,
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, y_pred, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, y_probability)),
}

with open(EVALUATION_DIR / "evaluation_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print(f"\nEvaluation artifacts saved to: {EVALUATION_DIR}")

print("\nMODEL EVALUATION")
print("-" * 40)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_probability):.3f}")

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Good Credit", "Bad Credit"],
    zero_division=0,
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Good", "Predicted Bad"],
    yticklabels=["Actual Good", "Actual Bad"],
)
plt.title("Loan Default Prediction - Confusion Matrix")
plt.xlabel("Model Prediction")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probability)
auc = roc_auc_score(y_test, y_probability)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Loan Default Prediction")
plt.legend()
plt.tight_layout()
plt.show()