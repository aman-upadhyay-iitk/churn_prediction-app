"""
train_and_save_model.py
------------------------
Trains the churn prediction pipeline (same as analysis.py) and saves it
as a .pkl file that the Streamlit app loads for live predictions.

Run this once (or whenever you re-run analysis.py with new/real data):
    python3 train_and_save_model.py
"""

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("telco_churn.csv")

features_num = ["tenure_months", "monthly_charges", "total_charges", "num_support_tickets"]
features_cat = ["contract", "internet_service", "payment_method", "tech_support",
                 "online_security", "partner", "paperless_billing"]

X = df[features_num + features_cat]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat)
], remainder="passthrough")

model = RandomForestClassifier(n_estimators=300, max_depth=8,
                                class_weight="balanced", random_state=42)

pipe = Pipeline([("prep", preprocessor), ("clf", model)])
pipe.fit(X_train, y_train)

with open("churn_model.pkl", "wb") as f:
    pickle.dump(pipe, f)

acc = pipe.score(X_test, y_test)
print(f"Model trained and saved -> churn_model.pkl (test accuracy: {acc:.3f})")
