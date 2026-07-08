# Online softmax recurrence

The single algorithmic idea at the core of Flash Attention. Compute `softmax(x)` over a stream of chunks, maintaining running state, without ever materializing the whole vector in memory.

## Problem

Given a vector `x` of length `N`, compute `softmax(x)` **chunk by chunk**, maintaining a running max and running denominator across chunks. At the end, produce the same output as `torch.softmax(x, dim=-1)` to `1e-6`.

Signature:

```python
def online_softmax(x: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    x: (N,) — the input vector
    chunk_size: process the input in chunks of this size (last chunk may be smaller)
    Returns: (N,) — the softmax over the whole vector
    """
    ...
```

Pure numpy. No PyTorch. No streaming API. Just implement the recurrence.

## The math

Standard stable softmax needs two passes over `x`:

```
m = max(x)                      # pass 1: reduce
d = sum(exp(x - m))             # pass 2: reduce
softmax(x) = exp(x - m) / d
```

**Online softmax computes m and d in a single pass over chunks.**

For each new chunk `xc`, maintain `(m_running, d_running)`:

```
m_new = max(xc)                                            # local chunk max
m_combined = max(m_running, m_new)                         # updated running max
d_running = d_running * exp(m_running - m_combined) + \    # rescale old denom
            sum(exp(xc - m_combined))                       # add new contribution
m_running = m_combined
```

**Why the rescale?** When the running max grows, everything you'd previously computed as `exp(x_old - m_running_old)` was assuming the OLD max. To use the NEW max instead, you multiply by `exp(m_running_old - m_running_new)`. Every previously-accumulated exponential gets rescaled uniformly.

After all chunks:

```
softmax(x_i) = exp(x_i - m_final) / d_final
```

Same shape (N,) output.

## Two-pass structure (simplest to get right first)

1. **Pass 1** — iterate chunks, maintain `(m_running, d_running)`. After all chunks, you have the final `m` and `d`.
2. **Pass 2** — iterate chunks again, compute `out[i] = exp(x_i - m) / d`.

Both passes visit the data in chunk-sized bites. The whole vector is never materialized in a single reduction op.

## True single-pass (harder, matches Flash Attention)

Flash Attention does everything in ONE pass by also maintaining a running numerator (the partial `exp(x - m) / d` values) and rescaling it every time `m` changes. Skip this on the first attempt; get the two-pass version right first.

## Test

```python
np.random.seed(0)
x = np.random.randn(1024)
out_online = online_softmax(x, chunk_size=64)

# Reference: standard softmax
m = np.max(x)
out_ref = np.exp(x - m) / np.exp(x - m).sum()

print("shapes match:", out_online.shape == out_ref.shape)
print("values match:", np.allclose(out_online, out_ref, atol=1e-6))
print("max diff:", np.abs(out_online - out_ref).max())
print("sums to 1:", np.isclose(out_online.sum(), 1.0))
```

All True → correct.

## Things to get right

- **Handle the last chunk** if `N % chunk_size != 0`. The last chunk is smaller than `chunk_size`.
- **Initial state**: `m_running = -inf`, `d_running = 0`. First chunk's update handles the "started from nothing" case naturally.
- **Numerical stability**: subtracting the running max keeps `exp()` bounded. Never call `exp` on a raw `x_i` without subtracting a max.
- **Rescaling** happens BEFORE adding new chunk's contribution. Order matters.

## Common bugs

1. **Skipping the rescale**: adding new chunk's `sum(exp(xc - m_combined))` to old `d_running` without multiplying old by `exp(m_old - m_combined)`. Results will be wrong whenever the max updates.
2. **Rescale in the wrong direction**: `exp(m_combined - m_old)` grows old d instead of shrinking. Always `exp(m_old - m_new)` where the argument is negative when max is growing.
3. **Not subtracting max in exp**: `exp(xc)` blows up. Always `exp(xc - m_combined)`.
4. **Off-by-one on the last chunk**: `x[i:i+chunk_size]` at the end. Fine in numpy; sanity check the last iteration.
5. **Two-pass version storing intermediates from pass 1 that pass 2 doesn't need**: unnecessary memory. Pass 2 just needs `m_final` and `d_final` — both scalars.

## What to say aloud (the Flash Attention connection)

- Standard softmax needs to materialize `x` twice: once to find max, once to compute the denominator. That's O(T²) memory traffic in attention where `x` is one row of the T×T attention matrix.
- **Flash Attention tiles Q, K, V into blocks and uses this recurrence to combine partial softmaxes across tiles without materializing the full T×T attention matrix.**
- Memory complexity of attention drops from O(T²) to O(T·d).
- FLOP count is roughly unchanged. What changes is HBM traffic — from O(T²) to O(T·d).
- LLM inference is memory-bound, so this is a big win. FA-2 is ~2× faster than vanilla attention on modern GPUs.

## References (after timer)

- FlashAttention paper (Dao et al., 2022) — Algorithm 1
- FlashAttention-2 paper (Dao, 2023) — swaps inner loop order for better GPU utilization
- Milakov & Gimelshein 2018, "Online normalizer calculation for softmax" — the original streaming softmax paper

## Attempt header template

```python
# Attempt N — YYYY-MM-DD
# Started:
# First pass (runs without error):
# Correct (allclose vs torch.softmax to 1e-6):
# Stuck on:
# Notes:
```
