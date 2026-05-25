# Day 3 — 2026-05-25

**Theme:** LC day — Top K Frequent Elements + Adam deep-dive

## Deep-dive (30 min target): `adam`

- Started: 3:40pm
- First pass (runs without error): 4:30pm (~50 min) — but failed correctness
- Correct (matches `torch.optim.Adam` over 50 steps to 1e-5): 4:55pm
- Total wall: ~1h15m (with hints)
- **PB set:** 1h15m baseline. Next attempt target: <20 min unaided.
- Architecture:
  - Per-param `m`, `v` dicts (lazy init on first `step()`).
  - Step counter `t` increments after each `step()`.
  - Bias correction via temp `m_hat`, `v_hat` — not stored back into state.
  - Update: `p -= lr * m_hat / (sqrt(v_hat) + eps)` (PyTorch convention: eps outside sqrt).
- Verification: 50-step training comparison vs `torch.optim.Adam` on identical seeded models → True True on both params.
- Bugs caught + fixes (detailed in file header):
  - Forgot the parameter update entirely on first pass — computed m,v but never touched p.
  - Bias correction was inverted (multiplied by `(1 - β^t)` instead of divided) AND stored back into state, corrupting the EMA.
  - Missing `self.lr` in the update — Adam took effective unit steps; params drifted to ~10.
  - Python scoping bug in test loop: `for model, opt in [[mlp, opt], [mlp_ref, opt_ref]]` reassigned `opt` to `opt_ref` after first inner iter; renamed inner var to `opti`.
  - MSELoss arg order was `(y, out)` — convention is `(input, target)`.
- Followups for attempt 2:
  - Zero-init `m`, `v` in `__init__` via dict comprehension; drops the if/else branch.

## LC (30 min target): Top K Frequent Elements (LC 347)

- Started: 5:37pm
- First pass (runs without error): 6:06pm (~29 min)
- Correct (passes 5 tests incl. edge cases): ~1h total
- **PB set:** 1h baseline. Next attempt target: <15 min unaided.
- Two implementations:
  - `topKFrequent`: incremental hash-of-hashes (`freq_to_num` + `num_to_freq`). Update both as we iterate `nums`. Switched to sets for O(1) avg remove.
  - `topKFrequent_bucket_sort`: canonical bucket sort (`Counter` + buckets indexed by freq, iterate from high to low).
- Verification: 5 tests including edge cases (single bucket > k, intermediate-empty buckets, all-same input). Both impls pass all.
- Bugs caught + fixes:
  - Loop counted buckets, not items — returned an entire top bucket regardless of k.
  - Empty intermediate buckets advanced `i`, returning fewer than k items.
  - Trim was inside the loop — never fired when only one bucket existed.
  - Bucket sort sized `len(nums)` not `len(nums) + 1` — IndexError when all numbers identical.
- Followups for attempt 2:
  - Default to bucket sort under timer pressure — incremental approach is harder to get right when bugs only surface on edge cases.

## Retro (one line)

Today's lesson: visual inspection of a passing test means nothing if the test doesn't exercise the bug. Both Adam and Top K Frequent had bugs that the obvious test cases hid — only edge-case tests forced them out.
