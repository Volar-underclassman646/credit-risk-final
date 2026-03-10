-- FILE: sql/03_etl_pipeline.sql
-- STEP 1: Deduplicate
DROP TABLE IF EXISTS cleaned_data.loan_deduped CASCADE;
CREATE TABLE cleaned_data.loan_deduped AS
SELECT DISTINCT ON (loan_amnt, annual_inc, issue_d, purpose, int_rate) *
FROM raw_data.loan_raw
ORDER BY loan_amnt, annual_inc, issue_d, purpose, int_rate, id;

-- STEP 2: Clean and transform
DROP TABLE IF EXISTS cleaned_data.loan_cleaned CASCADE;
CREATE TABLE cleaned_data.loan_cleaned AS
SELECT
    id, loan_amnt, funded_amnt, term, int_rate, installment,
    grade, sub_grade,
    CASE
        WHEN emp_length IS NULL THEN NULL
        WHEN emp_length = '< 1 year' THEN 0
        WHEN emp_length = '10+ years' THEN 10
        WHEN emp_length = '1 year' THEN 1
        ELSE CAST(REGEXP_REPLACE(emp_length,'[^0-9]','','g') AS INTEGER)
    END AS emp_length_years,
    home_ownership,
    COALESCE(annual_inc, (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY annual_inc) FROM cleaned_data.loan_deduped WHERE annual_inc IS NOT NULL)) AS annual_inc,
    verification_status, loan_status, purpose, addr_state,
    LEAST(COALESCE(dti, 0), 50) AS dti,
    delinq_2yrs,
    ROUND((COALESCE(fico_range_low,0) + COALESCE(fico_range_high,0))/2.0) AS fico_score,
    inq_last_6mths, open_acc, pub_rec, revol_bal,
    LEAST(COALESCE(revol_util, 0), 100) AS revol_util,
    total_acc, application_type
FROM cleaned_data.loan_deduped
WHERE loan_status IS NOT NULL AND loan_amnt > 0 AND annual_inc > 0;

-- STEP 3: Binary target
ALTER TABLE cleaned_data.loan_cleaned ADD COLUMN IF NOT EXISTS is_default INTEGER;
UPDATE cleaned_data.loan_cleaned SET is_default = CASE
    WHEN loan_status IN ('Charged Off','Default','Late (31-120 days)',
        'Does not meet the credit policy. Status:Charged Off') THEN 1
    ELSE 0 END;
