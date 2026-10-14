SELECT customer_id, churn_prob, risk_tier, rfm_segment, current_balance, occupation
FROM scored_customers
WHERE risk_tier = 'High'
ORDER BY current_balance DESC;
