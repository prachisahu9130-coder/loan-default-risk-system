from predict import predict_loan_risk


# Test applicant
applicant = {
    "checking_status": "A11",
    "duration": 12,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 2500,
    "savings_status": "A61",
    "employment": "A73",
    "installment_rate": 2,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "residence_since": 4,
    "property": "A121",
    "age": 35,
    "other_installment_plans": "A143",
    "housing": "A152",
    "existing_credits": 1,
    "job": "A173",
    "num_dependents": 1,
    "telephone": "A192",
    "foreign_worker": "A201"
}


# Make prediction
result = predict_loan_risk(applicant)


# Display result
print("=" * 50)
print("       LOAN RISK PREDICTION")
print("=" * 50)

print(f"Default Probability: {result['probability']:.2%}")
print(f"Risk Score: {result['risk_score']:.2f}/100")
print(f"Risk Level: {result['risk_level']}")
print(f"Classification: {result['classification']}")
print(f"Decision Threshold: {result['threshold']:.2f}")
print("\nTop Factors Influencing the Prediction:")

for factor in result["top_factors"]:
    print(f"Feature: {factor['feature']}")
    print(f"Contribution: {factor['contribution']:.4f}")
    print(f"Direction: {factor['direction']}")
    print("-" * 40)