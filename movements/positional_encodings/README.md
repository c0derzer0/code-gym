# Positional encodings

## Problem

Implement two variants from scratch:

### 1. Sinusoidal (original Transformer)

```python
def sinusoidal_pe(seq_len, d_model):
    """Returns (seq_len, d_model) tensor of fixed positional encodings."""
    ...
```

Even dims: `sin(pos / 10000^(2i/d_model))`; odd dims: `cos(...)`.

### 2. RoPE (Rotary)

```python
def apply_rope(q, k, seq_len):
    """
    q, k: (..., seq_len, d_head). Returns rotated q, k of same shape.
    """
    ...
```

RoPE rotates pairs of dims by an angle that depends on position. Apply *inside* attention (after Q,K projection, before the dot product). `V` is not rotated.

## Things to get right

- Sinusoidal: `10000^(2i/d_model)` = `exp(2i * -ln(10000) / d_model)`. Use the log form for numerical sanity.
- RoPE: split last dim into pairs, treat each pair as a 2D vector, rotate by `theta_i * pos`. Apply to Q and K only.
- Relative position is preserved by the inner product after RoPE rotation — that's the whole point.

## Test

- Sinusoidal: PE for position 0 should be `[0, 1, 0, 1, ...]`.
- RoPE: `<RoPE(q, m), RoPE(k, n)>` should depend only on `m - n`, not absolute positions. Sanity-check with `m=0, n=5` vs `m=2, n=7`.

## References (after timer)

- "Attention is All You Need" §3.5 (sinusoidal)
- RoFormer paper (RoPE)
- lucidrains `rotary-embedding-torch`
