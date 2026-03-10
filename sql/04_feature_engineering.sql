-- FILE: sql/04_feature_engineering.sql
DROP TABLE IF EXISTS analytics.loan_features CASCADE;
CREATE TABLE analytics.loan_features AS
SELECT *, 
    ROUND(loan_amnt / NULLIF(annual_inc,0), 4) AS lti_ratio,
    ROUND((installment*12) / NULLIF(annual_inc,0), 4) AS iti_ratio,
    ROUND(open_acc::DECIMAL / NULLIF(total_acc,0), 4) AS open_acc_ratio,
    CASE WHEN revol_util < 30 THEN 'Low' WHEN revol_util < 60 THEN 'Medium'
         WHEN revol_util < 80 THEN 'High' ELSE 'Very High' END AS credit_util_band,
    CASE WHEN annual_inc < 30000 THEN 'Very Low' WHEN annual_inc < 50000 THEN 'Low'
         WHEN annual_inc < 75000 THEN 'Medium' WHEN annual_inc < 100000 THEN 'High'
         ELSE 'Very High' END AS income_bracket,
    CASE grade WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 WHEN 'D' THEN 4
         WHEN 'E' THEN 5 WHEN 'F' THEN 6 WHEN 'G' THEN 7 ELSE 0 END AS grade_numeric,
    CASE WHEN pub_rec > 0 THEN 1 ELSE 0 END AS has_derog,
    CASE WHEN inq_last_6mths > 3 THEN 1 ELSE 0 END AS high_inquiry_flag,
    CASE WHEN delinq_2yrs > 0 THEN 1 ELSE 0 END AS has_delinq,
    CASE WHEN fico_score >= 750 THEN 'Excellent' WHEN fico_score >= 700 THEN 'Good'
         WHEN fico_score >= 650 THEN 'Fair' ELSE 'Poor' END AS fico_band,
    CASE WHEN grade IN ('A','B') AND dti < 15 AND fico_score >= 700 THEN 'Low Risk'
         WHEN grade IN ('C','D') OR dti BETWEEN 15 AND 30 THEN 'Medium Risk'
         ELSE 'High Risk' END AS risk_segment,
    CASE WHEN grade IN ('E','F','G') AND dti > 25 AND revol_util > 70 THEN 1 ELSE 0 END AS high_risk_flag
FROM cleaned_data.loan_cleaned;

CREATE INDEX idx_features_risk ON analytics.loan_features(risk_segment);
CREATE INDEX idx_features_default ON analytics.loan_features(is_default);
