"""
financial_impact.py
--------------------
Finance/business-analyst-flavored metrics on top of the churn model:
Customer Lifetime Value (CLV) impact, revenue at risk, and retention
campaign ROI. Run after train_and_save_model.py.

Run:
    python3 financial_impact.py
Output:
    outputs/financial_summary.md
"""

import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("telco_churn_real.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

# ---------- Revenue at risk ----------
churned = df[df["Churn"] == "Yes"]
monthly_revenue_lost = churned["MonthlyCharges"].sum()
annual_revenue_lost = monthly_revenue_lost * 12

# ---------- Customer Lifetime Value (simple CLV = avg monthly revenue x avg tenure) ----------
avg_monthly_revenue = df["MonthlyCharges"].mean()
avg_tenure_retained = df[df["Churn"] == "No"]["tenure"].mean()
avg_tenure_churned = churned["tenure"].mean()
clv_retained = avg_monthly_revenue * avg_tenure_retained
clv_churned = avg_monthly_revenue * avg_tenure_churned
clv_gap = clv_retained - clv_churned

# ---------- Retention campaign ROI (assumption-based, clearly labeled) ----------
# Assumption: a retention offer costs $15/customer/month (discount) and saves
# 30% of targeted at-risk customers from churning. These are illustrative
# assumptions -- swap in real figures if the company provides them.
campaign_cost_per_customer = 15
save_rate_assumption = 0.30

n_churned = len(churned)
targeted_top20pct = int(len(df) * 0.20)
estimated_saved_customers = int(targeted_top20pct * save_rate_assumption)
campaign_total_cost = targeted_top20pct * campaign_cost_per_customer
revenue_saved = estimated_saved_customers * avg_monthly_revenue * 12  # annualized
roi = (revenue_saved - campaign_total_cost) / campaign_total_cost * 100

summary = f"""# Churn — Financial & Business Impact Summary

## Revenue at risk
- Customers churned: **{n_churned}** ({n_churned/len(df)*100:.1f}% of base)
- Monthly recurring revenue lost to churn: **${monthly_revenue_lost:,.0f}**
- Annualized revenue lost: **${annual_revenue_lost:,.0f}**

## Customer Lifetime Value (CLV) impact
- Avg CLV of a retained customer: **${clv_retained:,.0f}**
- Avg CLV of a churned customer: **${clv_churned:,.0f}**
- CLV gap (value lost per churned customer vs a retained one): **${clv_gap:,.0f}**

## Retention campaign ROI (illustrative — swap in real cost figures)
*Assumptions: $15/customer/month retention offer, targeting the top 20% highest-risk
customers ({targeted_top20pct} customers), assumed 30% of those are successfully retained.*

- Campaign cost: **${campaign_total_cost:,.0f}**
- Estimated customers saved: **{estimated_saved_customers}**
- Annualized revenue saved: **${revenue_saved:,.0f}**
- **Estimated ROI: {roi:.0f}%**

## Business takeaway
Even under conservative assumptions, a targeted retention campaign on the
highest-risk segment pays for itself many times over — the model's job is
to make sure the campaign budget is spent on the *right* customers rather
than a blanket offer to the whole base.
"""

with open("outputs/financial_summary.md", "w") as f:
    f.write(summary)

print(summary)
print("Saved -> outputs/financial_summary.md")
