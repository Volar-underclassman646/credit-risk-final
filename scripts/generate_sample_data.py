"""
generate_sample_data.py - Generate Realistic Sample Data
==========================================================
YOU CANNOT PUT THE REAL LENDING CLUB CSV ON GITHUB.
(It's 200MB+ and has license restrictions)

Instead, this script generates realistic synthetic data.
Anyone who clones your repo runs this ONCE to get data.
This is the PROFESSIONAL way - same approach Google/Meta use.

Usage:
    python scripts/generate_sample_data.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_sample_data(n_rows=10000):
    """Generate synthetic lending data matching real Lending Club structure."""
    print(f"Generating {n_rows:,} synthetic loan records...")
    
    grades = np.random.choice(["A","B","C","D","E","F","G"], n_rows, p=[.15,.24,.26,.18,.10,.05,.02])
    sub_grades = [f"{g}{np.random.randint(1,6)}" for g in grades]
    
    grade_rate = {"A":.07,"B":.10,"C":.13,"D":.17,"E":.21,"F":.25,"G":.28}
    int_rates = np.clip([grade_rate[g]+np.random.normal(0,.015) for g in grades], .05, .31)
    
    loan_amnts = np.random.choice([5000,7500,10000,12000,15000,18000,20000,25000,30000,35000],
                                   n_rows, p=[.10,.08,.15,.12,.18,.10,.10,.08,.05,.04]).astype(float)
    terms = np.random.choice([" 36 months"," 60 months"], n_rows, p=[.72,.28])
    term_m = np.where(terms==" 36 months", 36, 60)
    mr = np.array(int_rates)/12
    installments = loan_amnts * (mr*(1+mr)**term_m) / ((1+mr)**term_m - 1)
    
    annual_inc = np.clip(np.random.lognormal(10.9, 0.6, n_rows), 12000, 500000).round(2)
    emp_lengths = np.random.choice(["< 1 year","1 year","2 years","3 years","4 years","5 years",
                                     "6 years","7 years","8 years","9 years","10+ years"],
                                    n_rows, p=[.08,.06,.08,.07,.06,.08,.06,.06,.06,.05,.34])
    home = np.random.choice(["RENT","MORTGAGE","OWN","OTHER"], n_rows, p=[.40,.44,.14,.02])
    verif = np.random.choice(["Verified","Not Verified","Source Verified"], n_rows, p=[.32,.35,.33])
    purposes = np.random.choice(["debt_consolidation","credit_card","home_improvement","other",
                                  "major_purchase","small_business","medical","car","moving",
                                  "vacation","house","wedding","renewable_energy","educational"],
                                 n_rows, p=[.47,.13,.07,.07,.05,.05,.04,.03,.02,.02,.02,.01,.01,.01])
    
    dti = np.clip(np.random.normal(18, 8, n_rows), 0, 50).round(2)
    fico_low = np.clip(np.random.normal(700, 35, n_rows).astype(int), 620, 845)
    fico_high = fico_low + 4
    open_acc = np.clip(np.random.poisson(11, n_rows), 1, 40)
    total_acc = np.clip(open_acc + np.random.poisson(10, n_rows), open_acc, 80)
    revol_bal = np.clip(np.random.lognormal(9.0, 1.2, n_rows), 0, 200000).round(2)
    revol_util = np.clip(np.random.normal(52, 25, n_rows), 0, 100).round(1)
    pub_rec = np.random.choice([0,1,2,3], n_rows, p=[.82,.12,.04,.02])
    delinq = np.random.choice([0,1,2,3], n_rows, p=[.75,.15,.07,.03])
    inq = np.random.choice([0,1,2,3,4,5], n_rows, p=[.40,.28,.16,.09,.04,.03])
    
    # Default probability correlated with risk factors
    grade_def = {"A":.06,"B":.11,"C":.17,"D":.25,"E":.34,"F":.41,"G":.49}
    prob = np.clip(np.array([grade_def[g] for g in grades]) + 
                   np.clip((dti-15)*.003, -.05,.10) +
                   np.clip((680-fico_low)*.001, -.05,.10) +
                   np.clip((revol_util-50)*.001, -.03,.05), .02, .70)
    is_def = (np.random.random(n_rows) < prob).astype(int)
    
    status = np.where(is_def==1, np.random.choice(["Charged Off","Default","Late (31-120 days)"],
                      n_rows, p=[.7,.15,.15]),
                      np.random.choice(["Fully Paid","Current"], n_rows, p=[.65,.35]))
    states = np.random.choice(["CA","NY","TX","FL","IL","PA","OH","GA","NC","NJ",
                                "VA","MA","WA","AZ","CO","MN","MO","IN","TN","MD"], n_rows)
    app_type = np.random.choice(["Individual","Joint App"], n_rows, p=[.86,.14])
    
    df = pd.DataFrame({
        "loan_amnt": loan_amnts, "funded_amnt": loan_amnts, "term": terms,
        "int_rate": np.round(int_rates,4), "installment": installments.round(2),
        "grade": grades, "sub_grade": sub_grades, "emp_length": emp_lengths,
        "home_ownership": home, "annual_inc": annual_inc,
        "verification_status": verif, "loan_status": status,
        "purpose": purposes, "addr_state": states, "dti": dti,
        "delinq_2yrs": delinq, "fico_range_low": fico_low, "fico_range_high": fico_high,
        "inq_last_6mths": inq, "open_acc": open_acc, "pub_rec": pub_rec,
        "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
        "application_type": app_type,
    })
    
    print(f"Done! Default rate: {is_def.mean()*100:.1f}%")
    return df

if __name__ == "__main__":
    df = generate_sample_data(10000)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/lending_club_sample.csv", index=False)
    print(f"Saved: data/raw/lending_club_sample.csv ({os.path.getsize('data/raw/lending_club_sample.csv')/1024:.0f} KB)")
    print("\nFor the REAL (120K+) dataset: download from Kaggle and save as data/raw/lending_club.csv")
