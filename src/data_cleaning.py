"""
data_cleaning.py - Clean and preprocess loan data
===================================================
Handles both PostgreSQL and CSV-based data loading.
Usage: python src/data_cleaning.py
"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")
load_dotenv()


def load_from_csv(path=None):
    if path is None:
        for p in ["data/raw/lending_club.csv", "data/raw/lending_club_sample.csv"]:
            if os.path.exists(p):
                path = p
                break
    if path is None or not os.path.exists(path):
        print("ERROR: No data file found. Run: python scripts/generate_sample_data.py")
        exit(1)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def create_target(df):
    defaults = ["Charged Off", "Default", "Late (31-120 days)",
                "Does not meet the credit policy. Status:Charged Off"]
    df["is_default"] = df["loan_status"].isin(defaults).astype(int)
    print(f"Default rate: {df['is_default'].mean()*100:.2f}%")
    return df


def select_features(df):
    keep = ["loan_amnt", "funded_amnt", "term", "int_rate", "installment",
            "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
            "verification_status", "purpose", "dti", "delinq_2yrs",
            "fico_range_low", "fico_range_high", "inq_last_6mths",
            "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
            "application_type", "loan_status", "is_default"]
    return df[[c for c in keep if c in df.columns]].copy()


def clean_emp_length(df):
    if "emp_length" not in df.columns:
        return df
    mapping = {"< 1 year": 0, "1 year": 1, "10+ years": 10}
    df["emp_length_years"] = df["emp_length"].map(mapping)
    mask = df["emp_length_years"].isna() & df["emp_length"].notna()
    df.loc[mask, "emp_length_years"] = df.loc[mask, "emp_length"].str.extract(r"(\d+)", expand=False).astype(float)
    df.drop(columns=["emp_length"], inplace=True, errors="ignore")
    return df


def clean_term(df):
    if "term" in df.columns:
        df["term_months"] = df["term"].str.extract(r"(\d+)", expand=False).astype(float)
        df.drop(columns=["term"], inplace=True, errors="ignore")
    return df


def create_fico_score(df):
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        df.drop(columns=["fico_range_low", "fico_range_high"], inplace=True)
    return df


def handle_missing(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def cap_outliers(df):
    skip = ["is_default", "has_derog", "high_risk_flag", "has_delinq", "high_inquiry_flag"]
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in skip:
            continue
        Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(Q1, Q3)
    return df


def remove_leakage(df):
    leak = ["total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
            "total_rec_late_fee", "recoveries", "collection_recovery_fee",
            "last_pymnt_d", "last_pymnt_amnt", "last_credit_pull_d",
            "last_fico_range_high", "last_fico_range_low", "out_prncp",
            "out_prncp_inv", "loan_status"]
    df.drop(columns=[c for c in leak if c in df.columns], inplace=True)
    return df


def validate(df):
    checks = {
        "Binary target": df["is_default"].isin([0, 1]).all(),
        "No nulls": df.isnull().sum().sum() == 0,
        "Rows > 10K": len(df) > 10000,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(checks.values())


def run_full_pipeline(source="csv"):
    print("="*60 + "\nDATA CLEANING PIPELINE\n" + "="*60)
    df = load_from_csv()
    df = create_target(df)
    df = select_features(df)
    df = clean_emp_length(df)
    df = clean_term(df)
    df = create_fico_score(df)
    df = df.dropna(axis=1, thresh=len(df)*0.6)
    df = handle_missing(df)
    df = cap_outliers(df)
    df = remove_leakage(df)
    validate(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_loans.csv", index=False)
    print(f"Saved: data/processed/cleaned_loans.csv ({df.shape})")
    return df


if __name__ == "__main__":
    run_full_pipeline()
