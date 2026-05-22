# Product of Array Except Self

LeetCode 238 · Medium

## Problem

Return an array `answer` where `answer[i]` = product of all elements of `nums` except `nums[i]`. Must run in O(n) and **without** division.

```python
def productExceptSelf(nums: list[int]) -> list[int]:
    ...
```

## Approach

Two passes:
- Left pass: `out[i]` = product of all elements to the left of `i`.
- Right pass: multiply by product of all elements to the right of `i`.

O(1) extra space (output array doesn't count).

## Test

- `[1,2,3,4]` → `[24,12,8,6]`
- `[-1,1,0,-3,3]` → `[0,0,9,0,0]`
- Watch zeros — division-based approach breaks on a single zero.
