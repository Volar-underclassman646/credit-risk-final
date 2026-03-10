# Data Dictionary

## Raw Features (Lending Club Dataset)
| Column | Type | Description |
|--------|------|-------------|
| loan_amnt | Numeric | Loan amount ($) |
| int_rate | Numeric | Interest rate (decimal) |
| grade | Categorical | LC grade A-G |
| annual_inc | Numeric | Annual income ($) |
| dti | Numeric | Debt-to-income ratio |
| fico_range_low/high | Numeric | FICO score range |
| revol_util | Numeric | Revolving utilization % |
| loan_status | Categorical | TARGET source |

## Engineered Features (14 total)
| Feature | Business Justification |
|---------|----------------------|
| is_default | Binary target: 1=Default, 0=Paid |
| lti_ratio | Loan-to-Income — higher = riskier |
| iti_ratio | Monthly installment burden |
| dti_capped | DTI capped at 50 |
| grade_numeric | A=1 through G=7 |
| credit_util_band | Low/Medium/High/Very High |
| risk_segment | Composite: Low/Medium/High Risk |
| high_risk_flag | Worst-case composite indicator |
| has_derog | Has public records |
| high_inquiry_flag | >3 inquiries in 6 months |
