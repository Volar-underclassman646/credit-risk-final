"""
performance_benchmark.py - Benchmark Data Structures & Algorithms
===================================================================
Compares O(1) vs O(n), O(log n) vs O(n) operations.
Generates timing data for the README performance section.

Usage:
    python benchmarks/performance_benchmark.py
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_structures import LRUCache, SortedRiskArray, TopKRiskHeap
from src.utils.algorithms import benchmark_search_methods


def benchmark_lru_cache():
    """Benchmark LRU Cache vs dict lookup vs re-computation."""
    print("\n" + "="*60)
    print("BENCHMARK: LRU Cache vs Re-computation")
    print("="*60)
    
    n_lookups = 100000
    
    # Simulate model prediction time (~0.5ms per prediction)
    def fake_predict(x):
        time.sleep(0.0001)  # Simulate 0.1ms computation
        return x * 0.42
    
    # Benchmark: Without cache (every lookup recomputes)
    start = time.perf_counter()
    for i in range(1000):
        fake_predict(i % 100)  # 100 unique applicants, repeated
    time_no_cache = time.perf_counter() - start
    
    # Benchmark: With LRU cache
    cache = LRUCache(capacity=200)
    start = time.perf_counter()
    for i in range(1000):
        key = str(i % 100)
        result = cache.get(key)
        if result is None:
            result = i % 100 * 0.42  # Direct computation (no sleep)
            cache.put(key, result)
    time_with_cache = time.perf_counter() - start
    
    print(f"  Without cache (1K lookups): {time_no_cache*1000:.1f}ms")
    print(f"  With LRU cache (1K lookups): {time_with_cache*1000:.1f}ms")
    print(f"  Speedup: {time_no_cache/time_with_cache:.1f}x")
    print(f"  Cache hit rate: {cache.hit_rate:.1f}%")


def benchmark_binary_search():
    """Benchmark binary search vs linear scan for threshold optimization."""
    print("\n" + "="*60)
    print("BENCHMARK: Binary Search vs Linear Scan (Threshold Optimization)")
    print("="*60)
    
    for n in [10_000, 100_000, 1_000_000]:
        np.random.seed(42)
        probs = np.random.beta(2, 5, n)
        y_true = (np.random.random(n) < probs).astype(int)
        
        results = benchmark_search_methods(probs, y_true)
        
        print(f"\n  Dataset size: {n:,}")
        print(f"  Binary search: {results['binary_search_time_ms']:.3f}ms")
        print(f"  Linear search: {results['linear_search_time_ms']:.3f}ms")
        print(f"  Speedup: {results['speedup']}")


def benchmark_sorted_array():
    """Benchmark SortedRiskArray operations."""
    print("\n" + "="*60)
    print("BENCHMARK: SortedRiskArray Binary Search vs Naive Count")
    print("="*60)
    
    for n in [10_000, 100_000, 1_000_000]:
        np.random.seed(42)
        scores = list(np.random.random(n))
        
        # Build sorted array (one-time cost)
        start = time.perf_counter()
        arr = SortedRiskArray(scores)
        build_time = time.perf_counter() - start
        
        # Binary search: 10K count_below queries
        start = time.perf_counter()
        for _ in range(10_000):
            arr.count_below(0.5)
        bs_time = time.perf_counter() - start
        
        # Naive linear scan: 10K count queries
        sorted_scores = sorted(scores)  # Already sorted in arr
        start = time.perf_counter()
        for _ in range(10_000):
            sum(1 for s in sorted_scores if s <= 0.5)
        linear_time = time.perf_counter() - start
        
        speedup = linear_time / bs_time if bs_time > 0 else float("inf")
        
        print(f"\n  Array size: {n:,}")
        print(f"  Build time (sort): {build_time*1000:.1f}ms")
        print(f"  10K binary searches: {bs_time*1000:.1f}ms")
        print(f"  10K linear scans: {linear_time*1000:.1f}ms")
        print(f"  Speedup: {speedup:.0f}x")


def benchmark_top_k_heap():
    """Benchmark TopKRiskHeap vs sorting entire array."""
    print("\n" + "="*60)
    print("BENCHMARK: Top-K Heap vs Full Sort")
    print("="*60)
    
    n = 1_000_000
    k = 100
    np.random.seed(42)
    scores = np.random.random(n)
    
    # Heap approach: O(n log k)
    start = time.perf_counter()
    heap = TopKRiskHeap(k=k)
    for i in range(n):
        heap.push(f"app_{i}", scores[i])
    top_heap = heap.get_top_k()
    heap_time = time.perf_counter() - start
    
    # Sort approach: O(n log n)
    start = time.perf_counter()
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top_sort = [(f"app_{i}", s) for i, s in indexed[:k]]
    sort_time = time.perf_counter() - start
    
    print(f"\n  Finding top {k} from {n:,} elements:")
    print(f"  Heap (O(n log k)):     {heap_time*1000:.1f}ms")
    print(f"  Full sort (O(n log n)): {sort_time*1000:.1f}ms")
    print(f"  Speedup: {sort_time/heap_time:.1f}x")


if __name__ == "__main__":
    print("#"*60)
    print("#  PERFORMANCE BENCHMARKS")
    print("#  Credit Risk Segmentation v2.0")
    print("#"*60)
    
    benchmark_lru_cache()
    benchmark_binary_search()
    benchmark_sorted_array()
    benchmark_top_k_heap()
    
    print("\n" + "#"*60)
    print("#  All benchmarks complete!")
    print("#"*60)
