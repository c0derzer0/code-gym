# Attempt 1 — 25 May, 2026
# Started: 5:37pm
# finished (first pass runs): 6:06pm (~29 min)
# finished after correction hints: ~1h total
# First pass (runs without error): yes
# Correct (passes 5 tests including edge cases): yes — both impls
# Stuck on: original impl had two correctness bugs only the edge cases exposed
# Notes (bugs caught + fixes, with hints):
#   - bug 1 (orig): loop counted buckets, not items — returned an entire top bucket
#     regardless of k. fix: count items added, not buckets visited.
#   - bug 2 (orig): empty buckets advanced `i`, so an intermediate-empty case
#     ([1,1,1,2], k=2) returned [1] instead of [1, 2]. fix: use len(top_k) < k.
#   - bug 3 (orig): trim was inside the loop — when only one bucket exists, the
#     trim never fires. fix: move `top_k[:k]` outside the loop.
#   - bug 4 (bucket): buckets sized `len(nums)` not `len(nums) + 1`. crashes when
#     a single number has freq == n (e.g., [1,1,1], k=1). fix: range(len(nums) + 1).
#   - swapped list→set for freq_to_num buckets to get O(1) avg remove instead of O(n).
# Both impls now pass all 5 tests. Canonical bucket sort is cleaner in an interview —
# incremental hash-of-hashes is harder to get right under timer.

from collections import defaultdict, Counter

def topKFrequent(nums: list[int], k: int) -> list[int]:
    freq_to_num = {}
    num_to_freq = defaultdict(int)

    for num in nums:
        prev_freq = num_to_freq[num]
        curr_freq = prev_freq + 1
        num_to_freq[num] = curr_freq

        if prev_freq != 0:
            freq_to_num[prev_freq].remove(num)

        if curr_freq in freq_to_num:
            freq_to_num[curr_freq].add(num)
        else:
            freq_to_num[curr_freq] = set()
            freq_to_num[curr_freq].add(num)
 
    #print(freq_to_num)
    top_k = []
    for i, (freq, nums) in enumerate(reversed(freq_to_num.items())):
        if len(top_k) < k:
            top_k.extend(nums)
        
    top_k = top_k[:k]

    return top_k
        

def topKFrequent_bucket_sort(nums: list[int], k: int) -> list[int]:
    counts = Counter(nums)
    #print(counts)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in counts.items():
        buckets[freq].append(num)
    #print(buckets)
    result = []
    for bucket in reversed(buckets):
        #print(bucket)
        for num in bucket:
            result.append(num)
            if len(result) == k:
                return result
                
    
nums = [1,1,1,2,2,3]
k = 2
top_k = topKFrequent_bucket_sort(nums, k)
print(top_k)
