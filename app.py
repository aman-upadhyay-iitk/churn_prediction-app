"""
app.py
------
Streamlit app: Customer Churn Prediction — live demo.
Enter a customer's details and get an instant churn risk prediction.

Run locally:
    pip install streamlit pandas scikit-learn
    streamlit run app.py

Deploy for free (so you get a shareable link for your resume):
    1. Push this whole folder to a public GitHub repo
    2. Go to https://share.streamlit.io -> "New app"
    3. Connect your GitHub repo, set main file = app.py
    4. Deploy -> you'll get a URL like https://your-app.streamlit.app
    5. Put that URL on your resume next to this project
"""

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Prediction Demo", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction")
st.write(
    "Enter a telecom customer's details below to predict their likelihood "
    "of churning, using a Random Forest model trained on customer usage "
    "and billing data."
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
    tenure_months = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                     float(round(monthly_charges * max(tenure_months, 1), 2)))
    num_support_tickets = st.slider("Support Tickets (last year)", 0, 10, 1)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

if st.button("Predict Churn Risk", type="primary"):
    input_df = pd.DataFrame([{
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_tickets": num_support_tickets,
        "contract": contract,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "tech_support": tech_support,
        "online_security": online_security,
        "partner": partner,
        "paperless_billing": paperless_billing
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
        if contract == "Month-to-month":
            notes.append("Month-to-month contracts have the highest churn rate.")
        if tenure_months < 12:
            notes.append("Low tenure customers are historically higher risk.")
        if payment_method == "Electronic check":
            notes.append("Electronic check payers show elevated churn in this dataset.")
        if tech_support == "No" or online_security == "No":
            notes.append("Lack of tech support / online security correlates with higher churn.")
        if num_support_tickets >= 3:
            notes.append("Multiple support tickets suggest dissatisfaction.")
        if not notes:
            notes.append("This customer's profile matches historically low-risk patterns.")
        for n in notes:
            st.write(f"- {n}")

st.divider()
st.caption(
    "Model: Random Forest Classifier trained on telecom customer data. "
    "Built as a portfolio project — see the GitHub repo for the full "
    "analysis pipeline (EDA, model comparison, SQL queries)."
)
