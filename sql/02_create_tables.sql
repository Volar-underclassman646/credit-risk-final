-- FILE: sql/02_create_tables.sql
DROP TABLE IF EXISTS raw_data.loan_raw CASCADE;
CREATE TABLE raw_data.loan_raw (
    id SERIAL PRIMARY KEY,
    loan_amnt DECIMAL(12,2), funded_amnt DECIMAL(12,2),
    term VARCHAR(20), int_rate DECIMAL(6,4), installment DECIMAL(10,2),
    grade VARCHAR(2), sub_grade VARCHAR(3), emp_title VARCHAR(100),
    emp_length VARCHAR(20), home_ownership VARCHAR(20),
    annual_inc DECIMAL(14,2), verification_status VARCHAR(30),
    issue_d VARCHAR(20), loan_status VARCHAR(50), purpose VARCHAR(30),
    title VARCHAR(100), zip_code VARCHAR(10), addr_state VARCHAR(5),
    dti DECIMAL(8,4), delinq_2yrs INTEGER,
    earliest_cr_line VARCHAR(20),
    fico_range_low INTEGER, fico_range_high INTEGER,
    inq_last_6mths INTEGER, open_acc INTEGER, pub_rec INTEGER,
    revol_bal DECIMAL(14,2), revol_util DECIMAL(6,2),
    total_acc INTEGER, initial_list_status VARCHAR(2),
    out_prncp DECIMAL(12,2), out_prncp_inv DECIMAL(12,2),
    total_pymnt DECIMAL(14,2), total_pymnt_inv DECIMAL(14,2),
    total_rec_prncp DECIMAL(14,2), total_rec_int DECIMAL(14,2),
    total_rec_late_fee DECIMAL(10,2), recoveries DECIMAL(12,2),
    collection_recovery_fee DECIMAL(10,2),
    last_pymnt_d VARCHAR(20), last_pymnt_amnt DECIMAL(12,2),
    last_credit_pull_d VARCHAR(20),
    last_fico_range_high INTEGER, last_fico_range_low INTEGER,
    application_type VARCHAR(20),
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_loan_status ON raw_data.loan_raw(loan_status);
CREATE INDEX idx_grade ON raw_data.loan_raw(grade);
