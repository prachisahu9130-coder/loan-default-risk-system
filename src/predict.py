
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# PATHS AND MODEL LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "loan_risk_model.pkl"
SHAP_DIR = BASE_DIR / "models" / "shap"
EXPLAINER_PATH = SHAP_DIR / "shap_explainer.pkl"

model = joblib.load(MODEL_PATH)
shap_explainer = joblib.load(EXPLAINER_PATH)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BEST_THRESHOLD = 0.35
TOP_N_FEATURES = 5


# --------------------------------------------------
# HELPER: MAKE FEATURE NAMES READABLE
# --------------------------------------------------

def readable_feature_name(feature_name):
    name = str(feature_name)

    for prefix in ("num__", "cat__"):
        name = name.replace(prefix, "")

    name = name.replace("_", " ").strip()

    if name.startswith("categorical "):
        name = name[len("categorical "):]

    if name.startswith("numerical "):
        name = name[len("numerical "):]

    return name.title()


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_loan_risk(applicant_data):

    # Convert input into a DataFrame
    applicant_df = pd.DataFrame([applicant_data])

    # Get probability of the bad-credit/default class
    probability = float(
        model.predict_proba(applicant_df)[0][1]
    )

    risk_score = probability * 100

    # ML classification using the selected threshold
    if probability >= BEST_THRESHOLD:
        classification = "Potential Default"
    else:
        classification = "Likely Good"

    # Separate presentation risk bands
    if risk_score < 30:
        risk_level = "Low Risk"
    elif risk_score < 60:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    # --------------------------------------------------
    # SHAP EXPLANATION FOR THIS APPLICANT
    # --------------------------------------------------

    preprocessor = model.named_steps["preprocessor"]

    transformed_applicant = preprocessor.transform(
        applicant_df
    )

    explanation = shap_explainer(transformed_applicant)
    print("Current applicant:")
    print(applicant_df)

    print("Current SHAP values:")
    print(explanation.values)

    shap_values = np.asarray(explanation.values)

    # One applicant: take its feature contributions
    applicant_shap_values = shap_values[0]

    # Get feature names from the fitted preprocessing pipeline
    feature_names = preprocessor.get_feature_names_out()

    # Sort features by absolute contribution
    top_indices = np.argsort(
        np.abs(applicant_shap_values)
    )[::-1][:TOP_N_FEATURES]

    top_factors = []

    for index in top_indices:

        contribution = float(applicant_shap_values[index])

        if contribution > 0:
            direction = "Increases predicted default risk"
        elif contribution < 0:
            direction = "Decreases predicted default risk"
        else:
            direction = "No directional contribution"

        top_factors.append({
            "feature": readable_feature_name(
                feature_names[index]
            ),
            "contribution": contribution,
            "direction": direction
        })

    # --------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------

    return {
        "probability": probability,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "classification": classification,
        "threshold": BEST_THRESHOLD,
        "top_factors": top_factors
    }