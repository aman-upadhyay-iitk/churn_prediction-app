-- ============================================================
-- churn_queries.sql
-- Standalone SQL business queries for the Telco Customer Churn
-- analysis. Import telco_churn_real.csv into any SQL engine
-- (SQLite, MySQL, PostgreSQL, or even Excel Power Query) as a
-- table named `customers` and run these directly.
-- ============================================================

-- 1. Overall churn rate
SELECT
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    COUNT(*) AS total_customers
FROM customers;

-- 2. Churn rate by contract type (biggest driver)
SELECT
    Contract,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. Churn rate by payment method
SELECT
    PaymentMethod,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 4. Tenure buckets vs churn (retention curve)
SELECT
    CASE
        WHEN tenure <= 6 THEN '0-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '48+ months'
    END AS tenure_bucket,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_bucket
ORDER BY MIN(tenure);

-- 5. Revenue at risk: monthly revenue currently held by churned customers
SELECT
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END) * 12, 2) AS annualized_revenue_lost
FROM customers;

-- 6. High-value at-risk segment: month-to-month, high monthly charges, churned
SELECT
    customerID, MonthlyCharges, tenure, Contract, PaymentMethod
FROM customers
WHERE Churn = 'Yes' AND Contract = 'Month-to-month' AND MonthlyCharges > 70
ORDER BY MonthlyCharges DESC
LIMIT 25;

-- 7. Internet service type vs churn
SELECT
    InternetService,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- 8. Senior citizens vs churn (demographic risk cut)
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS segment,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY segment;
