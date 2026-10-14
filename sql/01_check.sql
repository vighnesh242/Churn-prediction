SELECT COUNT(*) AS total, SUM(churn) AS churned,
       ROUND(AVG(churn) * 100, 1) AS churn_rate_pct
FROM customers;

SELECT MIN(last_transaction), MAX(last_transaction) FROM customers;

SELECT SUM(gender IS NULL) AS null_gender,
       SUM(occupation IS NULL) AS null_occupation,
       SUM(dependents IS NULL) AS null_dependents,
       SUM(last_transaction IS NULL) AS null_last_txn
FROM customers;
