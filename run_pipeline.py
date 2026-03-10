"""
run_pipeline.py - Run the complete project pipeline
Usage:
    python run_pipeline.py --csv    # Full pipeline from CSV
"""

import sys, os, time

def run_pipeline():
    start = time.time()
    print("#"*60 + "\n#  CREDIT RISK SEGMENTATION v2.0 - FULL PIPELINE\n" + "#"*60)
    
    # Step 0: Generate sample data if no data exists
    if not os.path.exists("data/raw/lending_club.csv") and not os.path.exists("data/raw/lending_club_sample.csv"):
        print("\n[0/5] GENERATING SAMPLE DATA")
        sys.path.insert(0, ".")
        from scripts.generate_sample_data import generate_sample_data
        df_sample = generate_sample_data(10000)
        os.makedirs("data/raw", exist_ok=True)
        df_sample.to_csv("data/raw/lending_club_sample.csv", index=False)
        print("Sample data generated!")
    
    print("\n[1/5] DATA CLEANING")
    from src.data_cleaning import run_full_pipeline
    df = run_full_pipeline()
    
    print("\n[2/5] FEATURE ENGINEERING")
    from src.feature_engineering import engineer_features, encode_and_prepare
    import pandas as pd
    df = engineer_features(df)
    X, y, df_dash = encode_and_prepare(df)
    pd.concat([X, y], axis=1).to_csv("data/processed/features_encoded.csv", index=False)
    df_dash.to_csv("data/processed/dashboard_data.csv", index=False)
    
    print("\n[3/5] MODEL TRAINING")
    from src.model_training import run_training
    best_model, X_test, y_test, y_prob = run_training()
    
    print("\n[4/5] POLICY SIMULATION")
    from src.policy_simulator import PolicySimulator
    import json
    test_df = pd.read_csv("data/processed/test_predictions.csv")
    amounts = test_df["loan_amnt"].values if "loan_amnt" in test_df.columns else None
    sim = PolicySimulator(test_df["y_true"].values, test_df["y_prob"].values, amounts)
    print(sim.run_simulation().to_string(index=False))
    sim.find_optimal()
    sim.plot_tradeoff()
    bench = sim.benchmark_vs_naive()
    os.makedirs("reports", exist_ok=True)
    with open("reports/policy_report.json", "w") as f:
        json.dump({"optimal": sim.find_optimal(), "benchmark": bench}, f, indent=2, default=str)
    
    print("\n[5/5] RUNNING TESTS")
    os.system("pytest tests/ -v --tb=short 2>/dev/null || echo 'Install pytest to run tests'")
    
    elapsed = time.time() - start
    print(f"\n{'#'*60}\n#  PIPELINE COMPLETE! ({elapsed/60:.1f} min)\n#  Run API: uvicorn src.api.scoring_api:app --port 8000\n{'#'*60}")

if __name__ == "__main__":
    run_pipeline()
