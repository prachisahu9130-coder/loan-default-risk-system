import joblib
import pandas as pd
import streamlit as st
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

# Add src folder to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.append(str(SRC_DIR))

from predict import predict_loan_risk


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Risk Assessment",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏦 Intelligent Loan Default Risk Assessment")

st.write(
    "Enter applicant information below to estimate "
    "the probability of loan default."
)

st.divider()


# --------------------------------------------------
# APPLICANT INFORMATION
# --------------------------------------------------

st.header("📋 Applicant Information")


col1, col2 = st.columns(2)


with col1:

    checking_status = st.selectbox(
        "Checking Account Status",
        ["A11", "A12", "A13", "A14"]
    )

    duration = st.number_input(
        "Loan Duration (months)",
        min_value=1,
        max_value=72,
        value=12
    )

    credit_history = st.selectbox(
        "Credit History",
        ["A30", "A31", "A32", "A33", "A34"]
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "A40",
            "A41",
            "A410",
            "A42",
            "A43",
            "A44",
            "A45",
            "A46",
            "A48",
            "A49"
        ]
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=250,
        max_value=20000,
        value=2500
    )

    savings_status = st.selectbox(
        "Savings Account Status",
        ["A61", "A62", "A63", "A64", "A65"]
    )

    employment = st.selectbox(
        "Employment",
        ["A71", "A72", "A73", "A74", "A75"]
    )

    installment_rate = st.slider(
        "Installment Rate (%)",
        min_value=1,
        max_value=4,
        value=2
    )

    personal_status_sex = st.selectbox(
        "Personal Status / Sex",
        ["A91", "A92", "A93", "A94", "A95"]
    )

    other_debtors = st.selectbox(
        "Other Debtors / Guarantors",
        ["A101", "A102", "A103"]
    )


with col2:

    residence_since = st.slider(
        "Years at Current Residence",
        min_value=1,
        max_value=4,
        value=4
    )

    property_status = st.selectbox(
        "Property",
        ["A121", "A122", "A123", "A124"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    other_installment_plans = st.selectbox(
        "Other Installment Plans",
        ["A141", "A142", "A143"]
    )

    housing = st.selectbox(
        "Housing",
        ["A151", "A152", "A153"]
    )

    existing_credits = st.number_input(
        "Number of Existing Credits",
        min_value=1,
        max_value=4,
        value=1
    )

    job = st.selectbox(
        "Job",
        ["A171", "A172", "A173", "A174"]
    )

    num_dependents = st.number_input(
        "Number of Dependents",
        min_value=1,
        max_value=10,
        value=1
    )

    telephone = st.selectbox(
        "Telephone",
        ["A191", "A192"]
    )

    foreign_worker = st.selectbox(
        "Foreign Worker",
        ["A201", "A202"]
    )


st.divider()


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button(
    "🔍 Assess Loan Risk",
    type="primary",
    use_container_width=True
):

    applicant = {

        "checking_status": checking_status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_status": savings_status,
        "employment": employment,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "residence_since": residence_since,
        "property": property_status,
        "age": age,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "num_dependents": num_dependents,
        "telephone": telephone,
        "foreign_worker": foreign_worker
    }


    # Make prediction
    result = predict_loan_risk(applicant)


    st.divider()

    st.header("📊 Loan Risk Assessment")


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Default Probability",
            f"{result['probability']:.2%}"
        )


    with col2:

        st.metric(
            "Risk Score",
            f"{result['risk_score']:.2f}/100"
        )


    with col3:

        st.metric(
            "Risk Level",
            result["risk_level"]
        )


    # Risk message
    if result["risk_level"] == "Low Risk":
        st.success("✅ Risk Band: LOW RISK")
    elif result["risk_level"] == "Medium Risk":
        st.warning("⚠️ Risk Band: MEDIUM RISK")
    else:
        st.error("🚨 Risk Band: HIGH RISK")

    # ML classification
    st.subheader("🤖 Machine Learning Classification")

    st.write(f"**Classification:** {result['classification']}")
    st.write(f"**Decision Threshold:** {result['threshold']:.0%}")

    st.progress(float(result["probability"]))

    st.info(
        "This educational model was trained on the German Credit "
        "dataset. Predictions are estimates, not guarantees of "
        "repayment or loan approval decisions."
    )

    
    
    st.subheader("🔎 Why did the model make this prediction?")

    st.caption(
        "Positive SHAP values push the model toward higher default risk; "
        "negative values push it toward lower default risk. Values are "
        "in the model's output scale, not percentage-point changes."
    )

    factors_df = pd.DataFrame(result["top_factors"])

    if not factors_df.empty:
        factors_df = factors_df.sort_values(
            by="contribution",
            ascending=True
        )

        factors_df["direction"] = factors_df["contribution"].apply(
            lambda value: (
                "Increases default risk" if value > 0
                else "Decreases default risk" if value < 0
                else "No directional contribution"
            )
        )

        chart_data = factors_df.set_index("feature")["contribution"]

        st.bar_chart(chart_data, horizontal=True)

        for _, factor in factors_df.sort_values(
            by="contribution",
            key=lambda column: column.abs(),
            ascending=False
        ).iterrows():
            st.markdown(f"**{factor['feature']}**")
            st.write(f"Contribution: {factor['contribution']:.4f}")
            st.write(f"Direction: {factor['direction']}")
            st.divider()

    st.caption(
        "These SHAP values explain which features influence the model's "
        "default-risk output. They are not causal explanations."
    )

    for factor in result["top_factors"]:
        st.markdown(f"**{factor['feature']}**")
        st.write(f"Contribution: {factor['contribution']:.4f}")
        st.write(f"Direction: {factor['direction']}")
        st.divider()
        



# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

import json
import numpy as np

st.divider()
st.header("📊 Model Performance")

st.write(
    "These charts evaluate the model on a held-out test dataset. "
    "They describe overall performance, not the current applicant."
)

EVALUATION_DIR = BASE_DIR / "models" / "evaluation"
Y_TEST_PATH = EVALUATION_DIR / "y_test.npy"
Y_PROBABILITY_PATH = EVALUATION_DIR / "y_probability.npy"
METRICS_PATH = EVALUATION_DIR / "evaluation_metrics.json"

if not all(
    path.exists()
    for path in [Y_TEST_PATH, Y_PROBABILITY_PATH, METRICS_PATH]
):
    st.warning(
        "Evaluation files are missing. Run "
        "`python src/evaluate_model.py` locally, then push the "
        "generated models/evaluation files to GitHub."
    )

else:
    y_test = np.load(Y_TEST_PATH)
    y_probability = np.load(Y_PROBABILITY_PATH)

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    threshold = metrics["threshold"]
    y_pred = (y_probability >= threshold).astype(int)

    st.caption(
        f"Classification threshold: {threshold:.0%} | "
        f"Test samples: {len(y_test)}"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    col3.metric("Precision (Bad Credit)", f"{metrics['precision']:.1%}")

    col1, col2 = st.columns(2)
    col1.metric("Recall (Bad Credit)", f"{metrics['recall']:.1%}")
    col2.metric("F1 Score (Bad Credit)", f"{metrics['f1_score']:.3f}")

    # Confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Good Credit", "Bad Credit"],
        yticklabels=["Good Credit", "Bad Credit"],
        ax=ax_cm,
    )

    ax_cm.set_xlabel("Predicted Class")
    ax_cm.set_ylabel("Actual Class")
    ax_cm.set_title("Actual vs Predicted Credit Class")

    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    st.caption(
        "Rows represent actual classes; columns represent predicted classes."
    )

    # ROC curve
    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_probability)

    fig_roc, ax_roc = plt.subplots(figsize=(7, 4))
    ax_roc.plot(
        fpr,
        tpr,
        label=f"Logistic Regression (AUC = {metrics['roc_auc']:.3f})",
    )
    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.25)

    fig_roc.tight_layout()
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    st.caption(
        "The ROC curve evaluates model discrimination across classification thresholds."
    )

