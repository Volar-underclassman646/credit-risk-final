"""
algorithms.py - Algorithm Implementations for Credit Risk
============================================================
Production-grade algorithms with documented time/space complexity.
Shows algorithmic thinking valued by Google/Microsoft/Amazon.

Includes:
    - Binary search for optimal threshold (O(log n) vs O(n) linear)
    - Merge sort for risk ranking (O(n log n) stable sort)
    - Two-pointer technique for paired risk analysis
    - Sliding window for time-series default rate monitoring
"""

import numpy as np
from typing import List, Tuple, Optional
import time


def binary_search_optimal_threshold(
    sorted_probs: np.ndarray,
    y_true: np.ndarray,
    sort_indices: np.ndarray,
    max_default_rate: float = 0.10,
    min_acceptance_rate: float = 0.40,
    precision: float = 0.001
) -> dict:
    """
    Binary search for the optimal approval threshold.
    
    Instead of testing every threshold from 0.05 to 0.95 (linear scan = O(n) per threshold),
    this uses binary search on the sorted probability array to find the optimal
    point in O(log(1/precision) * log(n)) time.
    
    Parameters
    ----------
    sorted_probs : np.ndarray
        Predicted default probabilities, SORTED ascending
    y_true : np.ndarray
        Actual default labels, reordered to match sorted_probs
    sort_indices : np.ndarray
        Indices that sort probs (from np.argsort)
    max_default_rate : float
        Maximum acceptable default rate among approved loans
    min_acceptance_rate : float
        Minimum acceptable acceptance rate
    precision : float
        Search precision for threshold
    
    Returns
    -------
    dict with optimal threshold and metrics
    
    Time Complexity: O(log(1/precision) * log(n))
    Space Complexity: O(n) for prefix sums
    """
    n = len(sorted_probs)
    
    # Build prefix sum of defaults for O(1) range queries
    # prefix_defaults[i] = number of defaults in sorted_probs[0:i]
    sorted_y = y_true[sort_indices]
    prefix_defaults = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        prefix_defaults[i + 1] = prefix_defaults[i] + sorted_y[i]
    
    def evaluate_threshold(threshold: float) -> dict:
        """Evaluate a threshold using binary search + prefix sum. O(log n)."""
        # Binary search for cutoff index
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_probs[mid] <= threshold:
                lo = mid + 1
            else:
                hi = mid
        
        approved_count = lo
        if approved_count == 0:
            return {"threshold": threshold, "acceptance_rate": 0,
                    "default_rate": 0, "approved": 0, "defaults": 0}
        
        defaults_in_approved = int(prefix_defaults[approved_count])
        acceptance_rate = approved_count / n
        default_rate = defaults_in_approved / approved_count
        
        return {
            "threshold": round(threshold, 4),
            "acceptance_rate": round(acceptance_rate, 4),
            "default_rate": round(default_rate, 4),
            "approved": approved_count,
            "defaults": defaults_in_approved,
        }
    
    # Binary search: find highest threshold where default_rate <= max_default_rate
    lo_t, hi_t = 0.0, 1.0
    best = None
    
    while hi_t - lo_t > precision:
        mid_t = (lo_t + hi_t) / 2
        result = evaluate_threshold(mid_t)
        
        if result["default_rate"] <= max_default_rate:
            if result["acceptance_rate"] >= min_acceptance_rate:
                best = result
            lo_t = mid_t  # Try higher threshold (more approvals)
        else:
            hi_t = mid_t  # Try lower threshold (fewer approvals)
    
    # If no valid threshold found, find closest
    if best is None:
        best = evaluate_threshold(lo_t)
        best["note"] = "No threshold meets both constraints"
    
    return best


def linear_search_optimal_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    max_default_rate: float = 0.10,
    step: float = 0.01
) -> dict:
    """
    Naive linear scan for comparison (baseline).
    
    Time Complexity: O(n * (1/step)) — much slower than binary search
    """
    n = len(probs)
    best = None
    
    threshold = 0.01
    while threshold <= 0.99:
        approved = probs <= threshold
        n_approved = approved.sum()
        
        if n_approved > 0:
            defaults = y_true[approved].sum()
            default_rate = defaults / n_approved
            acceptance_rate = n_approved / n
            
            if default_rate <= max_default_rate and acceptance_rate >= 0.40:
                if best is None or acceptance_rate > best["acceptance_rate"]:
                    best = {
                        "threshold": round(threshold, 4),
                        "acceptance_rate": round(acceptance_rate, 4),
                        "default_rate": round(default_rate, 4),
                        "approved": int(n_approved),
                        "defaults": int(defaults),
                    }
        
        threshold += step
    
    return best or {"note": "No valid threshold found"}


def sliding_window_default_rate(
    defaults: List[int],
    window_size: int = 30
) -> List[float]:
    """
    Sliding window to compute rolling default rate.
    
    Use Case: Monitor how default rate changes over time
    (e.g., 30-day rolling window for dashboard).
    
    Time Complexity: O(n) — single pass with running sum
    Space Complexity: O(n) for output array
    
    Parameters
    ----------
    defaults : List[int]
        Binary list of defaults ordered by time (0 or 1)
    window_size : int
        Rolling window size
    
    Returns
    -------
    List[float] : rolling default rate for each position
    """
    n = len(defaults)
    if n < window_size:
        return [sum(defaults) / n] if n > 0 else []
    
    rates = []
    window_sum = sum(defaults[:window_size])
    rates.append(window_sum / window_size)
    
    for i in range(window_size, n):
        window_sum += defaults[i] - defaults[i - window_size]
        rates.append(window_sum / window_size)
    
    return rates


def two_pointer_paired_risk(
    scores_a: List[float],
    scores_b: List[float],
    threshold: float
) -> List[Tuple[int, int]]:
    """
    Two-pointer technique to find pairs of applicants whose
    combined risk exceeds a threshold.
    
    Use Case: Co-borrower/joint loan risk analysis —
    find all pairs where combined risk is above a limit.
    
    Time Complexity: O(n log n) for sort + O(n) for two-pointer = O(n log n)
    Space Complexity: O(k) where k = number of pairs found
    """
    sorted_a = sorted(enumerate(scores_a), key=lambda x: x[1])
    sorted_b = sorted(enumerate(scores_b), key=lambda x: x[1], reverse=True)
    
    pairs = []
    j = 0
    
    for i in range(len(sorted_a)):
        while j < len(sorted_b) and sorted_a[i][1] + sorted_b[j][1] > threshold:
            pairs.append((sorted_a[i][0], sorted_b[j][0]))
            j += 1
        j = 0  # Reset for next element
    
    return pairs


def benchmark_search_methods(probs: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Benchmark binary search vs linear search for threshold optimization.
    Shows the performance difference with actual timing.
    """
    # Sort once for binary search
    sort_indices = np.argsort(probs)
    sorted_probs = probs[sort_indices]
    
    # Benchmark binary search
    start = time.perf_counter()
    for _ in range(100):
        result_bs = binary_search_optimal_threshold(
            sorted_probs, y_true, sort_indices
        )
    time_bs = (time.perf_counter() - start) / 100
    
    # Benchmark linear search
    start = time.perf_counter()
    for _ in range(100):
        result_ls = linear_search_optimal_threshold(probs, y_true)
    time_ls = (time.perf_counter() - start) / 100
    
    speedup = time_ls / time_bs if time_bs > 0 else float("inf")
    
    return {
        "binary_search_time_ms": round(time_bs * 1000, 3),
        "linear_search_time_ms": round(time_ls * 1000, 3),
        "speedup": f"{speedup:.1f}x faster",
        "binary_search_result": result_bs,
        "linear_search_result": result_ls,
        "data_size": len(probs),
    }
