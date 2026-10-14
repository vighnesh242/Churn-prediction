ALTER TABLE features ADD COLUMN rfm_segment TEXT;
UPDATE features SET rfm_segment = CASE
  WHEN r_score >= 4 AND m_score >= 4 THEN 'Champions'
  WHEN r_score >= 4 AND m_score <= 2 THEN 'Active Low Value'
  WHEN r_score <= 2 AND m_score >= 4 THEN 'At Risk High Value'
  WHEN r_score <= 2 AND m_score <= 2 THEN 'Dormant'
  ELSE 'Middle'
END;