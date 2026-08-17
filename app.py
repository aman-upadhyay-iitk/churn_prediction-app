"""
app.py  (REAL DATA VERSION)
----------------------------
Streamlit app: Customer Churn Prediction — live demo, trained on the
real Kaggle "Telco Customer Churn" dataset (7,043 customers).

Run locally:
    pip install streamlit pandas scikit-learn
    streamlit run app.py
"""

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Prediction Demo", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction")
st.write(
    "Enter a telecom customer's details below to predict their likelihood "
    "of churning, using a Random Forest model trained on the real Kaggle "
    "**Telco Customer Churn** dataset (7,043 customers)."
)

@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run `python3 train_and_save_model.py` first.")
    st.stop()

st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
    TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                    float(round(MonthlyCharges * max(tenure, 1), 2)))
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])

with col2:
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    Partner = st.selectbox("Has Partner", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

if st.button("Predict Churn Risk", type="primary"):
    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "Contract": Contract,
        "InternetService": InternetService,
        "PaymentMethod": PaymentMethod,
        "TechSupport": TechSupport,
        "OnlineSecurity": OnlineSecurity,
        "Partner": Partner,
        "PaperlessBilling": PaperlessBilling,
        "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
        "Dependents": Dependents,
        "MultipleLines": MultipleLines,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "PhoneService": PhoneService,
        "gender": gender
    }])

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Result")
    st.metric("Churn Probability", f"{prob*100:.1f}%")

    if pred == 1:
        st.error("⚠️ High churn risk — recommend adding this customer to the retention campaign.")
    else:
        st.success("✅ Low churn risk — customer likely to stay.")

    st.progress(min(int(prob * 100), 100))

    with st.expander("Why this prediction? (key risk factors)"):
        notes = []
        if Contract == "Month-to-month":
            notes.append("Month-to-month contracts have the highest churn rate.")
        if tenure < 12:
            notes.append("Low tenure customers are historically higher risk.")
        if PaymentMethod == "Electronic check":
            notes.append("Electronic check payers show elevated churn in this dataset.")
        if TechSupport == "No" or OnlineSecurity == "No":
            notes.append("Lack of tech support / online security correlates with higher churn.")
        if InternetService == "Fiber optic":
            notes.append("Fiber optic customers show higher churn than DSL customers.")
        if not notes:
            notes.append("This customer's profile matches historically low-risk patterns.")
        for n in notes:
            st.write(f"- {n}")

st.divider()
st.caption(
    "Model: Random Forest Classifier trained on the real Kaggle Telco "
    "Customer Churn dataset. Built as a portfolio project — see the "
    "GitHub repo for the full analysis pipeline (EDA, model comparison, "
    "SQL queries)."
)
