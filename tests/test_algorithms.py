"""
test_algorithms.py - Tests for algorithm implementations
=========================================================
Run: pytest tests/test_algorithms.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.algorithms import (
    binary_search_optimal_threshold,
    linear_search_optimal_threshold,
    sliding_window_default_rate,
    benchmark_search_methods,
)


class TestBinarySearchThreshold:
    """Test binary search for optimal threshold."""
    
    def setup_method(self):
        np.random.seed(42)
        self.n = 10000
        self.probs = np.random.beta(2, 5, self.n)
        self.y_true = (np.random.random(self.n) < self.probs).astype(int)
        self.sort_indices = np.argsort(self.probs)
        self.sorted_probs = self.probs[self.sort_indices]
    
    def test_finds_valid_threshold(self):
        result = binary_search_optimal_threshold(
            self.sorted_probs, self.y_true, self.sort_indices,
            max_default_rate=0.15, min_acceptance_rate=0.30
        )
        assert "threshold" in result
        assert result["acceptance_rate"] >= 0.30
        assert result["default_rate"] <= 0.15
    
    def test_matches_linear_search(self):
        result_bs = binary_search_optimal_threshold(
            self.sorted_probs, self.y_true, self.sort_indices,
            max_default_rate=0.15, min_acceptance_rate=0.30
        )
        result_ls = linear_search_optimal_threshold(
            self.probs, self.y_true, max_default_rate=0.15
        )
        # Both should find approximately the same acceptance rate
        if result_ls and "acceptance_rate" in result_ls:
            assert abs(result_bs["acceptance_rate"] - result_ls["acceptance_rate"]) < 0.05
    
    def test_strict_constraints_may_fail(self):
        result = binary_search_optimal_threshold(
            self.sorted_probs, self.y_true, self.sort_indices,
            max_default_rate=0.001, min_acceptance_rate=0.99
        )
        assert "note" in result or result["acceptance_rate"] < 0.99
    
    def test_binary_search_is_faster(self):
        results = benchmark_search_methods(self.probs, self.y_true)
        # Binary search should be faster
        assert results["binary_search_time_ms"] <= results["linear_search_time_ms"]


class TestSlidingWindow:
    """Test sliding window default rate."""
    
    def test_basic_window(self):
        defaults = [0, 0, 1, 0, 1, 1, 0, 0, 0, 1]
        rates = sliding_window_default_rate(defaults, window_size=5)
        assert len(rates) == 6  # n - window_size + 1
        assert rates[0] == 2/5  # first window: [0,0,1,0,1] has 2 defaults
    
    def test_all_defaults(self):
        defaults = [1, 1, 1, 1, 1]
        rates = sliding_window_default_rate(defaults, window_size=3)
        assert all(r == 1.0 for r in rates)
    
    def test_no_defaults(self):
        defaults = [0, 0, 0, 0, 0]
        rates = sliding_window_default_rate(defaults, window_size=3)
        assert all(r == 0.0 for r in rates)
    
    def test_window_larger_than_data(self):
        defaults = [1, 0, 1]
        rates = sliding_window_default_rate(defaults, window_size=10)
        assert len(rates) == 1
        assert abs(rates[0] - 2/3) < 0.01
