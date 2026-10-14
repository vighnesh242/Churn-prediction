DROP TABLE IF EXISTS features;
CREATE TABLE features AS
WITH ref AS (
  SELECT DATE(MAX(last_transaction), '+1 day') AS ref_date FROM customers
),
base AS (
  SELECT c.*,
    (c.last_transaction IS NOT NULL) AS has_last_txn,
    COALESCE(julianday(ref.ref_date) - julianday(c.last_transaction), 999) AS recency_days,
    (c.current_month_credit > 1) + (c.previous_month_credit > 1)
    + (c.current_month_debit > 1) + (c.previous_month_debit > 1) AS freq_score,
    c.current_month_credit + c.previous_month_credit
      + c.current_month_debit + c.previous_month_debit AS monetary_flow,
    c.current_balance - c.previous_month_end_balance AS balance_change,
    c.current_month_debit / NULLIF(c.previous_month_debit, 0) AS debit_trend
  FROM customers c CROSS JOIN ref
)
SELECT *,
  NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
  NTILE(5) OVER (ORDER BY monetary_flow) AS m_score
FROM base;