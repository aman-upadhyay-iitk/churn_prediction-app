# Customer Churn Prediction (Telecom)

An end-to-end ML project: EDA, model comparison, evaluation beyond
accuracy, and a business-framed retention-targeting recommendation —
the classic, most recognizable ML portfolio project for analyst/ML roles.

## What this does
- EDA on churn drivers: contract type, payment method, tenure
- Compares **Logistic Regression** and **Random Forest** (XGBoost auto-included
  if installed) for churn prediction
- Evaluates using **ROC-AUC, precision, recall** — not just accuracy
  (critical since churn is an imbalanced problem)
- Estimates real **business impact**: what % of actual churners you'd
  catch by targeting only the top 20% highest-risk customers
- Exports a ranked at-risk customer list + a Power BI-ready cleaned CSV

## How to run
```bash
pip install pandas numpy matplotlib scikit-learn
# optional: pip install xgboost   (script auto-detects and adds it)
python3 generate_data.py   # creates telco_churn.csv (synthetic, for testing)
python3 analysis.py        # runs the full pipeline -> outputs/
```

## ⚠️ Before you use this for your resume/portfolio
`generate_data.py` builds synthetic data with a genuine churn pattern
(driven by contract type, tenure, payment method, support tickets —
real factors, not random) so the model learns something real. For full
credibility on your resume:
1. Download the real **Kaggle "Telco Customer Churn"** dataset
   → https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Save it as `telco_churn.csv` (rename a couple of columns to match
   this script, or tweak the column names in `analysis.py`)
3. Re-run `python3 analysis.py`
4. (Optional, high-impact) Deploy a tiny Streamlit app where a recruiter
   can enter a customer's details and see the churn prediction live —
   great for a resume link/demo

## What to write on your resume
> Built a customer churn prediction pipeline in Python; compared
> Logistic Regression and Random Forest models (ROC-AUC 0.7+), and
> showed that targeting the top 20% highest-risk customers captures
> ~40% of actual churners — enabling efficient, targeted retention
> campaigns over blanket outreach.

## Live demo app (Streamlit)
This project includes a working web app where a recruiter can enter a
customer's details and get an instant churn prediction — this is a
strong resume differentiator (a clickable link beats a static screenshot).

**Run it locally:**
```bash
pip install streamlit pandas scikit-learn
python3 train_and_save_model.py   # trains and saves churn_model.pkl (run once)
streamlit run app.py              # opens in your browser
```

**Deploy it for free so you get a shareable link:**
1. Push this whole `churn_project/` folder to a public GitHub repo
2. Go to https://share.streamlit.io → "New app"
3. Connect your GitHub repo, set the main file to `app.py`
4. Deploy → you get a URL like `https://your-app.streamlit.app`
5. Put that URL on your resume and LinkedIn next to this project — recruiters
   can click and try it themselves, which stands out far more than a PDF screenshot

## Folder structure
```
churn_project/
├── generate_data.py         # synthetic data generator (swap for real Kaggle CSV)
├── analysis.py              # full pipeline: EDA -> ML -> evaluation -> business impact
├── train_and_save_model.py  # trains the model and saves churn_model.pkl for the app
├── app.py                   # Streamlit live-demo app
├── requirements.txt         # for local install / Streamlit Cloud deployment
├── churn_model.pkl          # trained model (regenerate after using real data)
├── README.md
└── outputs/
    ├── telco_churn_cleaned.csv
    ├── at_risk_customers_ranked.csv   (customers ranked by churn probability)
    ├── insights.md
    ├── model_metrics.md               (classification reports for both models)
    └── charts/                        (5 PNG visuals incl. ROC curve, confusion matrix)
```
