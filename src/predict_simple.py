import pandas as pd
import joblib
import os

print("="*60)
print("CREDIT SCORE PREDICTOR")
print("="*60)

# STEP 1: Load the trained model
print("\n📂 Step 1: Loading trained model...")
model = joblib.load('models/random_forest.pkl')
print("   ✅ Model loaded successfully")

# STEP 2: Create a sample applicant
print("\n👤 Step 2: Sample credit applicant...")

# Example: Someone applying for a loan
applicant = {
    'age': 35,
    'income': 65000,
    'num_credit_cards': 3,
    'years_credit_history': 10,
    'total_debt': 25000,
    'monthly_debt_payment': 1500,
    'num_late_payments_last_year': 1,
    'num_credit_inquiries': 2,
    'loan_amount': 20000
}

print("\nApplicant Details:")
for key, value in applicant.items():
    print(f"   {key:30s}: {value}")

# STEP 3: Make prediction
print("\n🔮 Step 3: Making prediction...")

# Convert to DataFrame (model expects this format)
applicant_df = pd.DataFrame([applicant])

# Predict
prediction = model.predict(applicant_df)[0]
probability = model.predict_proba(applicant_df)[0]

# STEP 4: Show results
print("\n" + "="*60)
print("PREDICTION RESULTS")
print("="*60)

if prediction == 1:
    print("⚠️  PREDICTION: Will DEFAULT (High Risk)")
    print(f"   Default Probability: {probability[1]*100:.1f}%")
    print("   ❌ RECOMMENDATION: REJECT loan application")
else:
    print("✅ PREDICTION: Will NOT default (Low Risk)")
    print(f"   Default Probability: {probability[1]*100:.1f}%")
    print("   ✅ RECOMMENDATION: APPROVE loan application")

print("="*60)

# STEP 5: Test multiple applicants
print("\n📊 Step 5: Testing multiple applicants...\n")

test_applicants = [
    {
        'name': 'Good Applicant',
        'age': 40, 'income': 90000, 'num_credit_cards': 4,
        'years_credit_history': 15, 'total_debt': 10000,
        'monthly_debt_payment': 800, 'num_late_payments_last_year': 0,
        'num_credit_inquiries': 1, 'loan_amount': 15000
    },
    {
        'name': 'Risky Applicant',
        'age': 25, 'income': 30000, 'num_credit_cards': 8,
        'years_credit_history': 2, 'total_debt': 50000,
        'monthly_debt_payment': 3000, 'num_late_payments_last_year': 8,
        'num_credit_inquiries': 12, 'loan_amount': 25000
    },
    {
        'name': 'Average Applicant',
        'age': 32, 'income': 55000, 'num_credit_cards': 3,
        'years_credit_history': 8, 'total_debt': 20000,
        'monthly_debt_payment': 1200, 'num_late_payments_last_year': 2,
        'num_credit_inquiries': 3, 'loan_amount': 18000
    }
]

for person in test_applicants:
    name = person.pop('name')
    person_df = pd.DataFrame([person])
    pred = model.predict(person_df)[0]
    prob = model.predict_proba(person_df)[0][1]
    
    status = "❌ HIGH RISK" if pred == 1 else "✅ LOW RISK"
    print(f"{name:20s} | Default Risk: {prob*100:5.1f}% | {status}")

print("\n" + "="*60)
print("✅ PREDICTION COMPLETE!")
print("="*60)
