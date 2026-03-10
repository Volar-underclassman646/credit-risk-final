"""
policy_simulator.py - Policy Simulation with Algorithmic Optimization
======================================================================
Uses binary search (O(log n)) instead of linear scan (O(n)) for
threshold optimization. Shows algorithmic thinking for Google/Microsoft.

Usage: python src/policy_simulator.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.data_structures import SortedRiskArray, RiskBucketMap
from src.utils.algorithms import binary_search_optimal_threshold, benchmark_search_methods


class PolicySimulator:
    """
    Optimized policy simulator using custom data structures.
    
    Key optimizations over naive approach:
    - SortedRiskArray: O(log n) threshold queries vs O(n) linear scan
    - Prefix sums: O(1) range default counts
    - Binary search: O(log(1/precision)) for optimal threshold
    """
    
    def __init__(self, y_true, y_prob, loan_amounts=None):
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)
        self.loan_amounts = np.array(loan_amounts) if loan_amounts is not None else None
        self.n = len(self.y_true)
        
        # Build optimized data structures
        self.sorted_array = SortedRiskArray(self.y_prob.tolist())
        self.sort_indices = np.argsort(self.y_prob)
        self.sorted_probs = self.y_prob[self.sort_indices]
        
        # Prefix sum for O(1) range queries
        sorted_y = self.y_true[self.sort_indices]
        self.prefix_defaults = np.zeros(self.n + 1, dtype=np.int64)
        for i in range(self.n):
            self.prefix_defaults[i+1] = self.prefix_defaults[i] + sorted_y[i]
        
        print(f"PolicySimulator: {self.n:,} applicants, {self.y_true.mean()*100:.1f}% default rate")
    
    def simulate_threshold(self, threshold):
        """O(log n) simulation using binary search + prefix sum."""
        approved_count = self.sorted_array.count_below(threshold)
        
        if approved_count == 0:
            return {"threshold": threshold, "acceptance_rate_pct": 0,
                    "default_rate_pct": 0, "approved": 0, "defaults": 0,
                    "default_reduction_pct": 0}
        
        defaults = int(self.prefix_defaults[approved_count])
        acceptance = approved_count / self.n
        default_rate = defaults / approved_count
        baseline = self.y_true.mean()
        reduction = (1 - default_rate / baseline) * 100 if baseline > 0 else 0
        
        return {
            "threshold": round(threshold, 3),
            "approved": approved_count,
            "acceptance_rate_pct": round(acceptance * 100, 2),
            "defaults": defaults,
            "default_rate_pct": round(default_rate * 100, 2),
            "default_reduction_pct": round(reduction, 2),
        }
    
    def run_simulation(self, start=0.05, stop=0.95, step=0.05):
        thresholds = np.arange(start, stop + step, step)
        return pd.DataFrame([self.simulate_threshold(t) for t in thresholds])
    
    def find_optimal(self, max_default_rate=0.10, min_acceptance_rate=0.40):
        """Find optimal threshold using O(log n) binary search."""
        result = binary_search_optimal_threshold(
            self.sorted_probs, self.y_true, self.sort_indices,
            max_default_rate, min_acceptance_rate
        )
        
        print(f"\nOptimal Policy (Binary Search):")
        print(f"  Threshold: {result.get('threshold', 'N/A')}")
        print(f"  Acceptance: {result.get('acceptance_rate', 0)*100:.1f}%")
        print(f"  Default Rate: {result.get('default_rate', 0)*100:.1f}%")
        return result
    
    def benchmark_vs_naive(self):
        """Compare binary search vs linear scan performance."""
        return benchmark_search_methods(self.y_prob, self.y_true)
    
    def plot_tradeoff(self, save="reports/figures/policy_tradeoff.png"):
        sim = self.run_simulation()
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.plot(sim["threshold"], sim["acceptance_rate_pct"], color="#3b82f6", linewidth=2.5, label="Acceptance %")
        ax1.set_xlabel("Threshold"); ax1.set_ylabel("Acceptance %", color="#3b82f6")
        ax2 = ax1.twinx()
        ax2.plot(sim["threshold"], sim["default_rate_pct"], color="#ef4444", linewidth=2.5, label="Default %")
        ax2.set_ylabel("Default %", color="#ef4444")
        plt.title("Policy Simulation: Acceptance vs Default Tradeoff")
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    df = pd.read_csv("data/processed/test_predictions.csv")
    amounts = df["loan_amnt"].values if "loan_amnt" in df.columns else None
    
    sim = PolicySimulator(df["y_true"].values, df["y_prob"].values, amounts)
    
    print("\nFull Simulation:")
    print(sim.run_simulation().to_string(index=False))
    
    sim.find_optimal()
    sim.plot_tradeoff()
    
    print("\nBenchmark: Binary Search vs Linear Scan:")
    bench = sim.benchmark_vs_naive()
    print(f"  Binary: {bench['binary_search_time_ms']:.3f}ms")
    print(f"  Linear: {bench['linear_search_time_ms']:.3f}ms")
    print(f"  Speedup: {bench['speedup']}")
    
    report = {"simulation": sim.run_simulation().to_dict(orient="records"),
              "optimal": sim.find_optimal(), "benchmark": bench}
    os.makedirs("reports", exist_ok=True)
    with open("reports/policy_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
