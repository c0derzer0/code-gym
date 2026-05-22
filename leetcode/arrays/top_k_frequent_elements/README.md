# Top K Frequent Elements

LeetCode 347 · Medium

## Problem

Given an array `nums` and integer `k`, return the `k` most frequent elements.

```python
def topKFrequent(nums: list[int], k: int) -> list[int]:
    ...
```

## Constraints

- `1 <= nums.length <= 10^5`
- `k` is in `[1, # unique elements]`
- Required: better than `O(n log n)`.

## Approaches (in order of merit)

1. **Bucket sort** — O(n). Count frequencies, place each value into a bucket indexed by frequency, walk buckets high → low until you have k.
2. **Heap of size k** — O(n log k). Counter, then `heapq.nsmallest(k, ...)` with negative counts.
3. **Sort** — O(n log n). Counter, sort by count desc, return top k. Acceptable but the interviewer will ask for better.

## Test

- `[1,1,1,2,2,3], k=2` → `[1,2]` (order doesn't matter).
- `[1], k=1` → `[1]`.
