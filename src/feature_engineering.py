"""
feature_engineering.py - Create risk-relevant features
=======================================================
Every feature has a documented business justification.
Usage: python src/feature_engineering.py
"""

import pandas as pd
import numpy as np
import os


def engineer_features(df):
    """Create all 14 engineered features."""
    df = df.copy()
    
    if "dti" in df.columns:
        df["dti_capped"] = df["dti"].clip(0, 50)
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["lti_ratio"] = (df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)).clip(0, 2)
        df["lti_ratio"].fillna(df["lti_ratio"].median(), inplace=True)
    if "installment" in df.columns and "annual_inc" in df.columns:
        df["iti_ratio"] = ((df["installment"]*12) / df["annual_inc"].replace(0, np.nan)).clip(0, 1)
        df["iti_ratio"].fillna(df["iti_ratio"].median(), inplace=True)
    if "revol_util" in df.columns:
        df["credit_util_band"] = pd.cut(df["revol_util"], bins=[-1,30,60,80,200], labels=["Low","Medium","High","Very High"])
    if "annual_inc" in df.columns:
        df["income_bracket"] = pd.cut(df["annual_inc"], bins=[0,30000,50000,75000,100000,float("inf")], labels=["Very Low","Low","Medium","High","Very High"])
    if "grade" in df.columns:
        df["grade_numeric"] = df["grade"].map({"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}).fillna(0).astype(int)
    if "open_acc" in df.columns and "total_acc" in df.columns:
        df["open_acc_ratio"] = (df["open_acc"] / df["total_acc"].replace(0, np.nan)).fillna(0.5)
    if "pub_rec" in df.columns:
        df["has_derog"] = (df["pub_rec"] > 0).astype(int)
    if "inq_last_6mths" in df.columns:
        df["high_inquiry_flag"] = (df["inq_last_6mths"] > 3).astype(int)
    if "delinq_2yrs" in df.columns:
        df["has_delinq"] = (df["delinq_2yrs"] > 0).astype(int)
    if "fico_score" in df.columns:
        df["fico_band"] = pd.cut(df["fico_score"], bins=[0,580,650,700,750,850], labels=["Very Poor","Fair","Good","Very Good","Excellent"])
    if all(c in df.columns for c in ["grade_numeric", "dti_capped"]):
        conditions = [(df["grade_numeric"]<=2) & (df["dti_capped"]<15), (df["grade_numeric"]<=4) | (df["dti_capped"].between(15,30))]
        df["risk_segment"] = np.select(conditions, ["Low Risk","Medium Risk"], default="High Risk")
    if all(c in df.columns for c in ["grade_numeric", "dti_capped", "revol_util"]):
        df["high_risk_flag"] = ((df["grade_numeric"]>=5) & (df["dti_capped"]>25) & (df["revol_util"]>70)).astype(int)
    
    print(f"Engineered 14 features. Total columns: {len(df.columns)}")
    return df


def encode_and_prepare(df):
    """Encode categoricals, return X, y, dashboard df."""
    drop = ["id","grade","sub_grade","loan_status","addr_state","emp_title","title","zip_code","earliest_cr_line","issue_d"]
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    
    df_dashboard = df.copy()
    cat_cols = [c for c in ["home_ownership","purpose","verification_status","application_type","credit_util_band","income_bracket","fico_band","risk_segment","loan_size_bucket"] if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    y = df_encoded["is_default"]
    X = df_encoded.drop(columns=["is_default"]).select_dtypes(include=[np.number])
    
    print(f"Features: {X.shape}, Target: {y.shape}, Default rate: {y.mean()*100:.2f}%")
    return X, y, df_dashboard


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_loans.csv")
    df = engineer_features(df)
    X, y, df_dash = encode_and_prepare(df)
    os.makedirs("data/processed", exist_ok=True)
    pd.concat([X, y], axis=1).to_csv("data/processed/features_encoded.csv", index=False)
    df_dash.to_csv("data/processed/dashboard_data.csv", index=False)
    print("Saved features and dashboard data.")
