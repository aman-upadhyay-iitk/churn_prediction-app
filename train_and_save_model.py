"""
train_and_save_model.py  (REAL DATA VERSION)
----------------------------------------------
Trains the churn prediction model on the REAL Kaggle "Telco Customer
Churn" dataset (renamed here to telco_churn_real.csv) and saves it as
churn_model.pkl for the Streamlit app.

Run:
    python3 train_and_save_model.py
"""

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("telco_churn_real.csv")

# TotalCharges has some blank strings in the real dataset -> convert & fill
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

features_num = ["tenure", "MonthlyCharges", "TotalCharges"]
features_cat = ["Contract", "InternetService", "PaymentMethod", "TechSupport",
                 "OnlineSecurity", "Partner", "PaperlessBilling",
                 "SeniorCitizen", "Dependents", "MultipleLines",
                 "OnlineBackup", "DeviceProtection", "StreamingTV",
                 "StreamingMovies", "PhoneService", "gender"]

X = df[features_num + features_cat]
y = df["Churn"]

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
print(f"Model trained on REAL data ({len(df)} customers) -> churn_model.pkl (test accuracy: {acc:.3f})")
