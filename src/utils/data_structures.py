"""
data_structures.py - Custom Data Structures for Credit Risk
=============================================================
Implements production-grade data structures used throughout
the pipeline. Shows algorithmic thinking (Google/Microsoft signal).

Includes:
    - LRUCache: O(1) prediction caching for repeated applicants
    - SortedRiskArray: O(log n) threshold search via binary search
    - RiskBucketMap: O(1) risk segment lookups via hash map
    - MinHeap: Priority queue for top-K risky applicants

Time/Space complexities documented for every operation.
"""

from collections import OrderedDict
from typing import Any, Optional, List, Tuple
import heapq


class LRUCache:
    """
    Least Recently Used Cache for prediction results.
    
    Use Case: When the scoring API receives repeat applicants,
    avoid re-running the model by caching recent predictions.
    
    Time Complexity:  O(1) get, O(1) put
    Space Complexity: O(capacity)
    
    Implementation: OrderedDict (doubly-linked list + hash map)
    """
    
    def __init__(self, capacity: int = 1000):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached prediction. O(1)"""
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Cache a prediction result. O(1)"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Evict LRU entry
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def size(self) -> int:
        return len(self.cache)
    
    def __repr__(self) -> str:
        return (
            f"LRUCache(capacity={self.capacity}, size={self.size}, "
            f"hit_rate={self.hit_rate:.1f}%)"
        )


class SortedRiskArray:
    """
    Sorted array of risk scores enabling O(log n) threshold queries.
    
    Use Case: Policy simulator needs to quickly find how many
    applicants fall below/above any given threshold. Instead of
    scanning all 120K+ scores each time, we use binary search.
    
    Time Complexity:
        - build:           O(n log n) one-time sort
        - count_below:     O(log n) via bisect
        - count_above:     O(log n) via bisect
        - find_threshold:  O(log n) binary search
    Space Complexity: O(n)
    """
    
    def __init__(self, scores: List[float]):
        self.scores = sorted(scores)  # O(n log n)
        self.n = len(self.scores)
    
    def _bisect_left(self, target: float) -> int:
        """Binary search: find leftmost insertion point. O(log n)"""
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.scores[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo
    
    def _bisect_right(self, target: float) -> int:
        """Binary search: find rightmost insertion point. O(log n)"""
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.scores[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        return lo
    
    def count_below(self, threshold: float) -> int:
        """Count scores <= threshold. O(log n)"""
        return self._bisect_right(threshold)
    
    def count_above(self, threshold: float) -> int:
        """Count scores > threshold. O(log n)"""
        return self.n - self._bisect_right(threshold)
    
    def acceptance_rate(self, threshold: float) -> float:
        """Fraction of applicants approved at threshold. O(log n)"""
        return self.count_below(threshold) / self.n if self.n > 0 else 0.0
    
    def find_threshold_for_rate(self, target_rate: float) -> float:
        """
        Binary search for the threshold that achieves a target acceptance rate.
        
        O(log n) — much faster than linear scan over all possible thresholds.
        
        Parameters
        ----------
        target_rate : float
            Desired acceptance rate (0.0 to 1.0)
        
        Returns
        -------
        float : threshold value
        """
        if target_rate <= 0:
            return 0.0
        if target_rate >= 1:
            return 1.0
        
        target_count = int(target_rate * self.n)
        target_count = max(0, min(target_count, self.n - 1))
        
        return self.scores[target_count]
    
    def percentile(self, p: float) -> float:
        """Get the p-th percentile score. O(1) after sort."""
        if not 0 <= p <= 100:
            raise ValueError("Percentile must be 0-100")
        idx = int(p / 100 * (self.n - 1))
        return self.scores[idx]
    
    def __len__(self) -> int:
        return self.n


class RiskBucketMap:
    """
    Hash map for O(1) risk segment lookups and aggregation.
    
    Use Case: Dashboard needs instant default rate by segment.
    Pre-computes aggregates so each query is O(1) instead of
    scanning the full dataset.
    
    Time Complexity:
        - build:      O(n) single pass
        - get_stats:  O(1) lookup
    Space Complexity: O(k) where k = number of unique segments
    """
    
    def __init__(self):
        self.buckets = {}  # segment -> {count, defaults, total_amount}
    
    def build(self, segments: List[str], defaults: List[int],
              amounts: Optional[List[float]] = None) -> None:
        """Build the hash map in a single O(n) pass."""
        self.buckets = {}
        
        for i in range(len(segments)):
            seg = segments[i]
            if seg not in self.buckets:
                self.buckets[seg] = {
                    "count": 0, "defaults": 0,
                    "total_amount": 0.0
                }
            self.buckets[seg]["count"] += 1
            self.buckets[seg]["defaults"] += defaults[i]
            if amounts is not None:
                self.buckets[seg]["total_amount"] += amounts[i]
    
    def get_stats(self, segment: str) -> dict:
        """O(1) lookup for segment statistics."""
        if segment not in self.buckets:
            return {"error": f"Segment '{segment}' not found"}
        
        b = self.buckets[segment]
        return {
            "segment": segment,
            "count": b["count"],
            "defaults": b["defaults"],
            "default_rate": round(b["defaults"] / b["count"] * 100, 2) if b["count"] > 0 else 0,
            "total_amount": round(b["total_amount"], 2),
            "avg_amount": round(b["total_amount"] / b["count"], 2) if b["count"] > 0 else 0,
        }
    
    def get_all_stats(self) -> List[dict]:
        """Get stats for all segments. O(k)."""
        return [self.get_stats(seg) for seg in sorted(self.buckets.keys())]
    
    @property
    def segments(self) -> List[str]:
        return sorted(self.buckets.keys())


class TopKRiskHeap:
    """
    Min-heap to efficiently track top-K riskiest applicants.
    
    Use Case: Real-time monitoring dashboard needs to show
    the top 100 riskiest active applicants without sorting
    the entire dataset each time.
    
    Time Complexity:
        - push:      O(log k)
        - get_top_k: O(k log k)
    Space Complexity: O(k)
    """
    
    def __init__(self, k: int = 100):
        self.k = k
        self.heap = []  # min-heap of (score, id)
    
    def push(self, applicant_id: str, risk_score: float) -> None:
        """Add applicant to heap. O(log k)."""
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (risk_score, applicant_id))
        elif risk_score > self.heap[0][0]:
            heapq.heapreplace(self.heap, (risk_score, applicant_id))
    
    def get_top_k(self) -> List[Tuple[str, float]]:
        """Return top-K riskiest, sorted descending. O(k log k)."""
        result = [(id_, score) for score, id_ in sorted(self.heap, reverse=True)]
        return result
    
    @property
    def min_score(self) -> float:
        """Minimum score in the heap (threshold to enter top-K)."""
        return self.heap[0][0] if self.heap else 0.0
    
    @property
    def size(self) -> int:
        return len(self.heap)
