"""
generate_data.py
-----------------
Generates a realistic synthetic telecom customer churn dataset that
mirrors the schema of Kaggle's famous "Telco Customer Churn" dataset
(tenure, contract type, monthly charges, services, churn flag), with a
genuine underlying churn pattern so the model learns something real.

NOTE: This is synthetic data for you to test the pipeline right now.
For your real resume/portfolio project, download the actual dataset:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Save it as telco_churn.csv (rename columns if needed) and re-run analysis.py.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 6000

tenure_months = np.random.randint(0, 73, n)
contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    n, p=[0.35, 0.20, 0.22, 0.23])
tech_support = np.random.choice(["Yes", "No"], n, p=[0.4, 0.6])
online_security = np.random.choice(["Yes", "No"], n, p=[0.35, 0.65])
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
num_support_tickets = np.random.poisson(1.2, n)

total_charges = np.round(monthly_charges * np.maximum(tenure_months, 1) *
                          np.random.uniform(0.95, 1.0, n), 2)

# genuine churn risk pattern (not random) driven by real factors
risk_score = (
    -2.3
    - 0.035 * tenure_months
    + np.where(contract == "Month-to-month", 1.1, np.where(contract == "One year", 0.2, -0.6))
    + 0.012 * monthly_charges
    + np.where(payment_method == "Electronic check", 0.5, 0)
    + np.where(tech_support == "No", 0.35, 0)
    + np.where(online_security == "No", 0.3, 0)
    + 0.15 * num_support_tickets
    + np.where(internet_service == "Fiber optic", 0.25, 0)
)
prob_churn = 1 / (1 + np.exp(-risk_score))
churn = (np.random.rand(n) < prob_churn).astype(int)

df = pd.DataFrame({
    "customer_id": [f"CUST{i:05d}" for i in range(1, n + 1)],
    "tenure_months": tenure_months,
    "contract": contract,
    "internet_service": internet_service,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "payment_method": payment_method,
    "tech_support": tech_support,
    "online_security": online_security,
    "senior_citizen": senior_citizen,
    "partner": partner,
    "paperless_billing": paperless_billing,
    "num_support_tickets": num_support_tickets,
    "churn": churn
})

df.to_csv("telco_churn.csv", index=False)
print(f"Generated {len(df)} rows -> telco_churn.csv")
print(f"Overall churn rate: {df['churn'].mean()*100:.2f}%")
