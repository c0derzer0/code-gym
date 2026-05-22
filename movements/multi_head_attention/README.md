# Multi-head attention

## Problem

Implement multi-head self-attention from scratch as an `nn.Module`. No `nn.MultiheadAttention`. Include support for a causal mask.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        ...
    def forward(self, x):
        """
        x: (B, T, d_model)
        Returns: (B, T, d_model)
        """
        ...
```

## Things to get right

- Split into heads via `reshape`/`view` to `(B, T, n_heads, d_head)` then transpose to `(B, n_heads, T, d_head)`. Process all heads in parallel.
- `d_head = d_model // n_heads`. Assert divisibility.
- Single `nn.Linear(d_model, 3 * d_model)` for Q/K/V projection is the GPT-style trick — faster than three separate.
- Causal mask: register a buffer (not parameter) for the lower-triangular mask.
- Final `nn.Linear(d_model, d_model)` output projection.

## Test

- Shape: input `(2, 16, 128)`, `n_heads=8` → output `(2, 16, 128)`.
- Causal: changing token at position `t` must not change output at positions `< t`.
- Param count: should be `4 * d_model^2 + 4 * d_model` (with biases).

## References (after timer)

- nanoGPT `CausalSelfAttention`
- "Attention is All You Need" §3.2.2

## Attempt header

```python
# Attempt N — YYYY-MM-DD
# Started:
# First pass:
# Correct:
# Stuck on:
# Notes:
```
