# Scaled dot-product attention

## Problem

Implement scaled dot-product attention from scratch in PyTorch. No `torch.nn.functional.scaled_dot_product_attention`, no `nn.MultiheadAttention`.

```python
def attention(Q, K, V, mask=None):
    """
    Q: (..., seq_q, d_k)
    K: (..., seq_k, d_k)
    V: (..., seq_k, d_v)
    mask: broadcastable to (..., seq_q, seq_k). True = keep, False = mask out.
    Returns: (..., seq_q, d_v)
    """
    ...
```

## Test (write your own in `attempts/`)

- Shape: random `Q,K,V` of shape `(2, 8, 64)` → output `(2, 8, 64)`.
- Mask: lower-triangular causal mask should zero out attention to future positions.
- Compare numerically to `torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)` — within `1e-5`.

## Things to get right

- Scale by `sqrt(d_k)`, not `d_k`.
- Mask before softmax (fill with `-inf`, then softmax).
- Numerically stable softmax (`torch.softmax` is already stable, but know why).

## References (after timer)

- "Attention is All You Need" §3.2.1
- Karpathy nanoGPT `CausalSelfAttention.forward`
- `torch.nn.functional.scaled_dot_product_attention` docs

## Attempt file header

```python
# Attempt N — YYYY-MM-DD
# Started:
# First pass (runs without error):
# Correct (passes tests):
# Stuck on:
# Notes:
```
