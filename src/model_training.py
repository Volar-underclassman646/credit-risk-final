"""
model_training.py - Train and evaluate ML models
==================================================
OOP design pattern with ModelTrainer class.
Usage: python src/model_training.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, f1_score, accuracy_score
import json, joblib, os, warnings

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

SEED = 42
FIG_DIR = "reports/figures"


class ModelTrainer:
    """Encapsulates the entire model training pipeline."""
    
    def __init__(self, X, y, test_size=0.2):
        os.makedirs(FIG_DIR, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=SEED, stratify=y)
        self.feature_names = X.columns.tolist()
        self.results = {}
        self.models = {}
        
        print(f"Train: {len(self.X_train):,} | Test: {len(self.X_test):,} | Default: {self.y_train.mean()*100:.1f}%")
    
    def _evaluate(self, name, model, X_test, y_pred, y_prob):
        auc = roc_auc_score(self.y_test, y_prob)
        f1 = f1_score(self.y_test, y_pred)
        acc = accuracy_score(self.y_test, y_pred)
        metrics = {"accuracy": acc, "f1": f1, "auc_roc": auc}
        self.results[name] = (y_prob, metrics)
        self.models[name] = model
        print(f"\n{name}: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")
        print(classification_report(self.y_test, y_pred))
        return metrics
    
    def train_logistic_regression(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        if HAS_SMOTE:
            X_bal, y_bal = SMOTE(random_state=SEED).fit_resample(X_scaled, self.y_train)
        else:
            X_bal, y_bal = X_scaled, self.y_train
        
        model = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        model.fit(X_bal, y_bal)
        joblib.dump(scaler, "models/scaler.pkl")
        
        return self._evaluate("Logistic Regression", model, X_test_scaled,
                              model.predict(X_test_scaled),
                              model.predict_proba(X_test_scaled)[:, 1])
    
    def train_random_forest(self):
        model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10,
                                        class_weight="balanced", random_state=SEED, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        return self._evaluate("Random Forest", model, self.X_test,
                              model.predict(self.X_test),
                              model.predict_proba(self.X_test)[:, 1])
    
    def train_xgboost(self):
        if not HAS_XGB:
            print("XGBoost not installed. Skipping.")
            return None
        scale_w = (self.y_train==0).sum() / (self.y_train==1).sum()
        model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               scale_pos_weight=scale_w, random_state=SEED,
                               eval_metric="auc", use_label_encoder=False)
        model.fit(self.X_train, self.y_train)
        return self._evaluate("XGBoost", model, self.X_test,
                              model.predict(self.X_test),
                              model.predict_proba(self.X_test)[:, 1])
    
    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))
        colors = {"Logistic Regression": "#3498DB", "Random Forest": "#2ECC71", "XGBoost": "#E74C3C"}
        for name, (y_prob, m) in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={m['auc_roc']:.3f})", color=colors.get(name), linewidth=2)
        plt.plot([0,1],[0,1],"k--", alpha=0.5)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("ROC Curve Comparison"); plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(f"{FIG_DIR}/roc_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
    
    def plot_shap(self):
        if not HAS_SHAP or "XGBoost" not in self.models:
            return
        explainer = shap.TreeExplainer(self.models["XGBoost"])
        shap_vals = explainer.shap_values(self.X_test[:1000])
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, self.X_test[:1000], feature_names=self.feature_names, show=False, max_display=15)
        plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    
    def select_and_save_best(self):
        best_name = max(self.results, key=lambda k: self.results[k][1]["auc_roc"])
        best = self.models[best_name]
        joblib.dump(best, "models/best_model.pkl")
        
        metrics = {n: m for n, (_, m) in self.results.items()}
        metrics["best_model"] = best_name
        with open("models/model_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save test predictions for policy simulator
        test_out = self.X_test.copy()
        test_out["y_true"] = self.y_test.values
        test_out["y_prob"] = self.results[best_name][0]
        test_out.to_csv("data/processed/test_predictions.csv", index=False)
        
        print(f"\nBest: {best_name} (AUC={self.results[best_name][1]['auc_roc']:.4f})")
        return best, self.X_test, self.y_test, self.results[best_name][0]


def run_training():
    print("="*60 + "\nMODEL TRAINING PIPELINE\n" + "="*60)
    df = pd.read_csv("data/processed/features_encoded.csv")
    y = df["is_default"]; X = df.drop(columns=["is_default"])
    
    trainer = ModelTrainer(X, y)
    trainer.train_logistic_regression()
    trainer.train_random_forest()
    trainer.train_xgboost()
    trainer.plot_roc_curves()
    trainer.plot_shap()
    return trainer.select_and_save_best()


if __name__ == "__main__":
    run_training()
