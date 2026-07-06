# KV cache

Extend `multi_head_attention` to cache K and V across autoregressive generation steps. The single biggest LLM-inference optimization.

## Problem

Modify your existing MHA class so `forward()` supports an optional cache. During autoregressive generation, K and V for past tokens have already been computed — cache them and only compute Q/K/V for the newly-arrived token.

```python
class MultiHeadAttnWithKVCache(nn.Module):
    def __init__(self, max_seq_len, d_model, n_heads, causal=True):
        ...

    def forward(self, x, cache=None):
        """
        x:     (B, T_new, d_model) — during generation, T_new == 1
        cache: {'k': (B, H, T_past, d_head), 'v': (B, H, T_past, d_head)}
               or None (first step)

        Returns: (out, new_cache)
          out:       (B, T_new, d_model)
          new_cache: {'k': (B, H, T_past + T_new, d_head), 'v': same shape}
        """
        ...
```

## Complexity — the load-bearing math

Without cache: at step `t`, recompute K, V for all `t` past tokens plus the new one. Per token cost O(t · d_model). Full generation of length T: O(T³ · d_model).

With cache: at step `t`, only compute Q, K, V for the ONE new token, append to cache, attend Q over the full cached K/V. Per token cost O(t · d_model) attention only. Full generation: O(T² · d_model). One factor of T saved.

**Memory cost of the cache itself:**
```
cache_bytes = 2 · L · n_kv_heads · T · d_head · sizeof(dtype)
```
For Llama-3-70B, T=4096, fp16: `2 · 80 · 8 · 4096 · 128 · 2 ≈ 1.3 GB` per sequence. **This is the dominant memory cost of serving.**

## Things to get right

- **Only compute Q/K/V for `x` (the new tokens).** Don't recompute K/V for past tokens — that's the whole point.
- **Concatenate along the sequence dim.** New K/V has shape `(B, H, T_new, d_head)`. Old cache has `(B, H, T_past, d_head)`. Concat along `dim=-2` → `(B, H, T_past + T_new, d_head)`.
- **Causal mask slicing.** Attention scores now have shape `(B, H, T_new, T_past + T_new)`. Slice your registered causal mask as `self.causal_mask[T_past : T_past + T_new, : T_past + T_new]`. The `T_new` rows correspond to the new tokens; each attends over past + own position.
- **`cache=None` first step.** When no cache is provided, this should behave exactly like your original MHA (full-sequence forward). Same output.
- **Return both `out` and `new_cache`.** The caller needs the updated cache for the next step.
- **`torch.no_grad()` context during generation.** Not needed for correctness of the movement, but relevant for real inference.

## Test — the correctness proof

The whole implementation is verified by one property: **generating one token at a time with the cache produces identical output to a full-sequence forward without the cache.**

```python
torch.manual_seed(0)
B, T, d_model, n_heads = 1, 10, 16, 4
mha = MultiHeadAttnWithKVCache(max_seq_len=10, d_model=d_model, n_heads=n_heads, causal=True)

x = torch.randn(B, T, d_model)

# Method 1: full-sequence forward, no cache
with torch.no_grad():
    out_full, _ = mha(x, cache=None)

# Method 2: one token at a time, with cache
cache = None
outs = []
with torch.no_grad():
    for t in range(T):
        x_t = x[:, t:t+1, :]
        out_t, cache = mha(x_t, cache=cache)
        outs.append(out_t)
out_cached = torch.cat(outs, dim=1)

assert out_full.shape == out_cached.shape
assert torch.allclose(out_full, out_cached, atol=1e-5)
assert cache['k'].shape == (B, n_heads, T, d_model // n_heads)
print("✓ full vs cached generation match to 1e-5")
```

If the values match, your KV cache is correct.

## Common bugs

1. **Recomputing K, V for past tokens.** Then the cache is dead weight. The whole point is that `k_new = self.k_proj(x)` only sees `x` (the new tokens), not the past.
2. **Concatenating along the wrong dim.** K/V have shape `(B, H, T, d_head)`. Concat along `dim=-2` (the T axis). Getting this wrong produces a shape error, or worse, a silent broadcast.
3. **Wrong causal mask slice.** Attention shape is `(B, H, T_new, T_past + T_new)`. The mask must be sliced to `[T_past : T_past + T_new, : T_past + T_new]`. Off-by-one here gives wrong outputs that still "look reasonable" — the correctness test catches it.
4. **Forgetting `T_past`.** Where do you get `T_past` from? From `cache['k'].shape[-2]`. Don't hardcode.
5. **Not handling `cache=None` cleanly.** First step has `T_past = 0`. Either special-case it or use `torch.zeros((B, H, 0, d_head))` as an empty cache to avoid branching.
6. **Not returning the updated cache.** Caller can't advance to the next step without it.

## What to say aloud (2-minute explanation)

- Without KV cache, generation is O(T³): each new token recomputes attention over the growing prefix, and there are T tokens.
- With KV cache, K and V for past tokens are stored and reused. Only Q for the new token, and its own K/V row, need to be computed. O(T²) total.
- The cache lives across the batch dim, per layer, per KV head. Memory: `2 · L · n_kv_heads · T · d_head · bytes`.
- For a 70B model at 4K context, that's ~1 GB per sequence — dominant memory cost at inference.
- This is why GQA exists (reduce n_kv_heads), why PagedAttention exists (reuse memory across padding), why sliding window attention exists (bound T in cache).

## References (after timer)

- nanoGPT — `model.py` has a minimal KV-cache-in-generation loop
- HuggingFace transformers `modeling_llama.py` — production-grade KV cache (dict of `past_key_values`)
- vLLM `PagedAttention` — the memory-management layer on top of this

## Attempt header template

```python
# Attempt N — YYYY-MM-DD
# Started:
# First pass (runs without error):
# Correct (cached generation matches full-sequence to 1e-5):
# Stuck on:
# Notes:
```
