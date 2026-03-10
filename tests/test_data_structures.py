"""
test_data_structures.py - Unit tests for custom data structures
================================================================
Covers edge cases, boundary conditions, and performance guarantees.
Run: pytest tests/test_data_structures.py -v
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_structures import (
    LRUCache, SortedRiskArray, RiskBucketMap, TopKRiskHeap
)


# ============================================================
# LRU CACHE TESTS
# ============================================================
class TestLRUCache:
    """Test LRU Cache with all edge cases."""
    
    def test_basic_put_get(self):
        cache = LRUCache(capacity=3)
        cache.put("a", 0.85)
        assert cache.get("a") == 0.85
    
    def test_cache_miss_returns_none(self):
        cache = LRUCache(capacity=3)
        assert cache.get("nonexistent") is None
    
    def test_eviction_on_capacity(self):
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
    
    def test_lru_order_updated_on_get(self):
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")       # "a" is now most recently used
        cache.put("c", 3)    # Should evict "b", not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
    
    def test_overwrite_existing_key(self):
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("a", 99)  # Update value
        assert cache.get("a") == 99
        assert cache.size == 1
    
    def test_hit_rate_tracking(self):
        cache = LRUCache(capacity=5)
        cache.put("a", 1)
        cache.get("a")    # hit
        cache.get("a")    # hit
        cache.get("b")    # miss
        assert cache.hits == 2
        assert cache.misses == 1
        assert abs(cache.hit_rate - 66.67) < 0.1
    
    def test_clear(self):
        cache = LRUCache(capacity=5)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None
    
    def test_capacity_one(self):
        cache = LRUCache(capacity=1)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") is None
        assert cache.get("b") == 2
    
    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=0)
        with pytest.raises(ValueError):
            LRUCache(capacity=-1)
    
    def test_performance_o1_operations(self):
        """Verify O(1) time complexity for large cache."""
        cache = LRUCache(capacity=100000)
        
        # Fill cache
        for i in range(100000):
            cache.put(str(i), i * 0.001)
        
        # Time 10000 gets — should be fast (O(1) each)
        start = time.perf_counter()
        for i in range(10000):
            cache.get(str(i + 50000))
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.1, f"10K gets took {elapsed:.3f}s — not O(1)"


# ============================================================
# SORTED RISK ARRAY TESTS
# ============================================================
class TestSortedRiskArray:
    """Test binary search on sorted risk scores."""
    
    def test_count_below(self):
        arr = SortedRiskArray([0.1, 0.2, 0.3, 0.4, 0.5])
        assert arr.count_below(0.3) == 3
        assert arr.count_below(0.0) == 0
        assert arr.count_below(1.0) == 5
    
    def test_count_above(self):
        arr = SortedRiskArray([0.1, 0.2, 0.3, 0.4, 0.5])
        assert arr.count_above(0.3) == 2
    
    def test_acceptance_rate(self):
        arr = SortedRiskArray([0.1, 0.2, 0.3, 0.4, 0.5])
        assert arr.acceptance_rate(0.3) == 0.6  # 3 out of 5
    
    def test_find_threshold_for_rate(self):
        scores = [i * 0.01 for i in range(100)]
        arr = SortedRiskArray(scores)
        threshold = arr.find_threshold_for_rate(0.50)
        # 50% acceptance = score at index 49
        assert abs(threshold - 0.49) < 0.02
    
    def test_percentile(self):
        arr = SortedRiskArray([10, 20, 30, 40, 50])
        assert arr.percentile(0) == 10
        assert arr.percentile(100) == 50
    
    def test_empty_array(self):
        arr = SortedRiskArray([])
        assert len(arr) == 0
        assert arr.acceptance_rate(0.5) == 0.0
    
    def test_binary_search_performance(self):
        """Verify O(log n) for large arrays."""
        import random
        scores = [random.random() for _ in range(1_000_000)]
        arr = SortedRiskArray(scores)
        
        start = time.perf_counter()
        for _ in range(10000):
            arr.count_below(0.5)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.1, f"10K binary searches on 1M elements took {elapsed:.3f}s"


# ============================================================
# RISK BUCKET MAP TESTS
# ============================================================
class TestRiskBucketMap:
    """Test hash map for segment lookups."""
    
    def test_build_and_lookup(self):
        bucket_map = RiskBucketMap()
        bucket_map.build(
            segments=["Low Risk", "Low Risk", "High Risk"],
            defaults=[0, 0, 1],
            amounts=[10000, 15000, 20000]
        )
        stats = bucket_map.get_stats("Low Risk")
        assert stats["count"] == 2
        assert stats["defaults"] == 0
        assert stats["default_rate"] == 0
    
    def test_default_rate_calculation(self):
        bucket_map = RiskBucketMap()
        bucket_map.build(
            segments=["High Risk"] * 10,
            defaults=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        )
        stats = bucket_map.get_stats("High Risk")
        assert stats["default_rate"] == 30.0
    
    def test_missing_segment(self):
        bucket_map = RiskBucketMap()
        bucket_map.build(segments=["A"], defaults=[0])
        result = bucket_map.get_stats("B")
        assert "error" in result
    
    def test_segments_list(self):
        bucket_map = RiskBucketMap()
        bucket_map.build(
            segments=["Low Risk", "High Risk", "Medium Risk"],
            defaults=[0, 1, 0],
        )
        assert bucket_map.segments == ["High Risk", "Low Risk", "Medium Risk"]


# ============================================================
# TOP-K RISK HEAP TESTS
# ============================================================
class TestTopKRiskHeap:
    """Test min-heap for top-K tracking."""
    
    def test_basic_top_k(self):
        heap = TopKRiskHeap(k=3)
        for i, score in enumerate([0.1, 0.5, 0.9, 0.3, 0.7]):
            heap.push(f"app_{i}", score)
        
        top = heap.get_top_k()
        assert len(top) == 3
        assert top[0][1] == 0.9  # Highest risk first
    
    def test_heap_size_limit(self):
        heap = TopKRiskHeap(k=2)
        for i in range(100):
            heap.push(f"app_{i}", i * 0.01)
        assert heap.size == 2
    
    def test_min_score_threshold(self):
        heap = TopKRiskHeap(k=3)
        heap.push("a", 0.5)
        heap.push("b", 0.7)
        heap.push("c", 0.9)
        assert heap.min_score == 0.5
    
    def test_empty_heap(self):
        heap = TopKRiskHeap(k=5)
        assert heap.size == 0
        assert heap.get_top_k() == []
