"""
app.py  (REAL DATA VERSION + SQL Insights tab)
------------------------------------------------
Streamlit app: Customer Churn Prediction — live demo, trained on the
real Kaggle "Telco Customer Churn" dataset (7,043 customers).
Includes a live SQL Insights tab (runs churn_queries.sql live via SQLite).

Run locally:
    pip install streamlit pandas scikit-learn
    streamlit run app.py
"""

import pickle
import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Prediction Demo", page_icon="📉", layout="wide")

st.title("📉 Customer Churn Prediction")
st.write(
    "Live demo trained on the real Kaggle **Telco Customer Churn** dataset "
    "(7,043 customers) — includes both an ML predictor and live SQL business insights."
)

tab1, tab2 = st.tabs(["🔮 Churn Predictor", "🗄️ SQL Insights"])

# ============================================================
# TAB 1: PREDICTOR
# ============================================================
with tab1:
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
        "Customer Churn dataset."
    )

# ============================================================
# TAB 2: SQL INSIGHTS (runs real SQL queries live, via SQLite)
# ============================================================
with tab2:
    st.subheader("Live SQL Business Queries")
    st.write(
        "These queries run **live** against the dataset using SQL (SQLite engine) — "
        "the same queries are also in `churn_queries.sql` for use in any SQL tool."
    )

    @st.cache_data
    def load_data():
        df = pd.read_csv("telco_churn_real.csv")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
        return df

    df = load_data()
    conn = sqlite3.connect(":memory:")
    df.to_sql("customers", conn, index=False, if_exists="replace")

    queries = {
        "Overall churn rate": """
            SELECT ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
                   COUNT(*) AS total_customers
            FROM customers;
        """,
        "Churn rate by contract type": """
            SELECT Contract, COUNT(*) AS customers,
                   ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
            FROM customers GROUP BY Contract ORDER BY churn_rate_pct DESC;
        """,
        "Churn rate by payment method": """
            SELECT PaymentMethod, COUNT(*) AS customers,
                   ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
            FROM customers GROUP BY PaymentMethod ORDER BY churn_rate_pct DESC;
        """,
        "Tenure buckets vs churn": """
            SELECT
                CASE WHEN tenure <= 6 THEN '0-6 months'
                     WHEN tenure <= 12 THEN '7-12 months'
                     WHEN tenure <= 24 THEN '13-24 months'
                     WHEN tenure <= 48 THEN '25-48 months'
                     ELSE '48+ months' END AS tenure_bucket,
                COUNT(*) AS customers,
                ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
            FROM customers GROUP BY tenure_bucket ORDER BY MIN(tenure);
        """,
        "Revenue at risk (churned customers)": """
            SELECT ROUND(SUM(CASE WHEN Churn='Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost,
                   ROUND(SUM(CASE WHEN Churn='Yes' THEN MonthlyCharges ELSE 0 END) * 12, 2) AS annualized_revenue_lost
            FROM customers;
        """,
        "Top 25 high-value at-risk customers": """
            SELECT customerID, MonthlyCharges, tenure, Contract, PaymentMethod
            FROM customers
            WHERE Churn='Yes' AND Contract='Month-to-month' AND MonthlyCharges > 70
            ORDER BY MonthlyCharges DESC LIMIT 25;
        """,
    }

    query_choice = st.selectbox("Choose a business question", list(queries.keys()))
    result = pd.read_sql_query(queries[query_choice], conn)

    st.code(queries[query_choice].strip(), language="sql")
    st.dataframe(result, use_container_width=True)

    # auto-chart for the 2-column grouped results
    if result.shape[1] == 3 and result.shape[0] > 1:
        chart_col = result.columns[0]
        value_col = result.columns[-1]
        st.bar_chart(result.set_index(chart_col)[value_col])
