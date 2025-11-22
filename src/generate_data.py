import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of people in our fake dataset
n_people = 1000

print("Creating fake credit data for practice...")

# Create fake data
data = {
    # Personal info
    'age': np.random.randint(20, 70, n_people),
    'income': np.random.randint(20000, 150000, n_people),
    
    # Credit history
    'num_credit_cards': np.random.randint(0, 10, n_people),
    'years_credit_history': np.random.randint(0, 40, n_people),
    
    # Debt information
    'total_debt': np.random.randint(0, 100000, n_people),
    'monthly_debt_payment': np.random.randint(0, 5000, n_people),
    
    # Payment behavior
    'num_late_payments_last_year': np.random.randint(0, 12, n_people),
    'num_credit_inquiries': np.random.randint(0, 10, n_people),
    
    # Loan info
    'loan_amount': np.random.randint(5000, 50000, n_people),
}

# Create DataFrame
df = pd.DataFrame(data)

# Create target variable (will they default?)
# People are more likely to default if they have:
# - Low income, high debt, many late payments
default_probability = (
    (df['total_debt'] / df['income']) * 0.3 +  # High debt-to-income ratio
    (df['num_late_payments_last_year'] / 12) * 0.4 +  # Many late payments
    (df['num_credit_inquiries'] / 10) * 0.2 -  # Too many credit checks
    (df['years_credit_history'] / 40) * 0.1  # Short credit history
).clip(0, 1)  # Keep between 0 and 1

# Randomly assign default based on probability
df['default'] = (np.random.random(n_people) < default_probability).astype(int)

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save to CSV
df.to_csv('data/credit_data.csv', index=False)

print(f"✅ Created {n_people} fake credit records")
print(f"✅ Saved to: data/credit_data.csv")
print(f"\nQuick look at the data:")
print(df.head())
print(f"\nDefault rate: {df['default'].mean()*100:.1f}%")
