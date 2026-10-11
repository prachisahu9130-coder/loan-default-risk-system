# 🏦 Intelligent Loan Default Prediction & Risk Scoring System

## 📌 Project Overview

The **Intelligent Loan Default Prediction & Risk Scoring System** is a machine learning application that estimates the probability of credit default based on applicant information.

Built using Python, Scikit-learn, Streamlit, and SHAP, the application predicts credit risk, displays model performance metrics, and explains the factors influencing individual predictions.

This project demonstrates the application of machine learning and Explainable AI (XAI) to credit-risk analysis.

## 🎯 Project Objectives

- Predict the likelihood of loan default using applicant data.
- Generate a numerical risk score based on the predicted default probability.
- Classify applicants into Low, Medium, or High Risk categories.
- Explain predictions using SHAP feature contributions.
- Evaluate model performance using standard classification metrics.
- Provide an interactive dashboard for exploring predictions and explanations.

## ✨ Key Features

- **Loan Default Prediction:** Estimates the probability of default using a trained Logistic Regression model.
- **Credit Risk Scoring:** Converts predicted probability into a score from 0 to 100.
- **Risk Classification:** Assigns a risk category based on the predicted probability.
- **Explainable AI:** Uses SHAP to identify factors that increase or decrease the model's predicted default risk.
- **Model Evaluation:** Displays Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- **Confusion Matrix:** Visualizes correct predictions and classification errors.
- **ROC Curve:** Evaluates the model's ability to distinguish between credit classes.
- **Interactive Dashboard:** Built with Streamlit for a user-friendly experience.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Scikit-learn | Model training, preprocessing, and evaluation |
| Logistic Regression | Classification algorithm |
| SHAP | Explainable AI |
| Streamlit | Interactive web application |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| Joblib | Saving and loading the trained model |

## 📊 Dataset

This project uses the **German Credit dataset** for educational model development and evaluation.

The target variable is mapped as follows:

- `0` — Good Credit
- `1` — Bad Credit

The dataset contains applicant characteristics and credit-related information.

The dataset is excluded from this repository. Obtain it from an authorized source and place the file at `data/german.data` before running the full application, subject to its applicable usage terms.

## 🤖 Model Performance

The current model evaluation on a held-out test set produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 77.5% |
| Precision (Bad Credit) | 59.7% |
| Recall (Bad Credit) | 76.7% |
| F1 Score (Bad Credit) | 0.672 |
| ROC-AUC | 0.804 |

The classification threshold is set to `0.35`.

These results reflect the current dataset split and evaluation setup. They do not guarantee performance on new applicants or real-world banking data.

## 🔍 Explainable AI (SHAP)

SHAP is used to explain how input features contribute to the model's predictions.

The application displays influential factors and indicates whether their contributions increase or decrease the model's predicted default risk.

SHAP contributions explain the behavior of the trained model; they do not establish causation.

## 📁 Project Structure

```text
loan-default-risk-system/
├── app/
│   └── app.py
├── data/
│   └── german.data
├── models/
│   ├── loan_risk_model.pkl
│   └── shap/
│       ├── feature_names.npy
│       ├── shap_explainer.pkl
│       └── shap_values.npy
├── notebooks/
│   └── 01_understanding_data.ipynb
├── src/
│   ├── evaluate_model.py
│   ├── predict.py
│   └── test_prediction.py
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/prachisahu9130-coder/loan-default-risk-system.git
cd loan-default-risk-system
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Obtain the German Credit dataset from an authorized source and place it at:

```text
data/german.data
```

The dataset is required by the current model-performance section of the Streamlit application.

### 5. Run the application

```bash
streamlit run app/app.py
```

The application will open in your browser.

## ⚠️ Limitations and Responsible Use

- This project is an educational prototype, not a production-ready credit decision system.
- Model predictions are estimates, not guarantees of repayment or default.
- Historical data may contain bias and may not represent current applicants.
- Credit decisions require broader validation, fairness assessment, privacy protection, security controls, and regulatory review.
- The model should not be used to approve or reject real loan applications without appropriate professional and institutional validation.

## 🚀 Future Improvements

- Deploy the application using Streamlit Community Cloud.
- Add automated model retraining and model monitoring.
- Improve data validation and error handling.
- Conduct fairness and robustness evaluations.
- Add automated tests and a reproducible training pipeline.
- Integrate a database for securely managed applicant records.

## 👩‍💻 Author

**Prachi Sahu**

Machine Learning | Artificial Intelligence | Data Analytics

GitHub: [prachisahu9130-coder](https://github.com/prachisahu9130-coder)

---

*Developed as a portfolio project to demonstrate machine learning, credit-risk scoring, model evaluation, and Explainable AI.*
