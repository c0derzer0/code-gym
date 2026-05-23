# Personal bests

Best **time-to-correct** per movement. "Correct" = passes the test block in the movement's README.

Update after each attempt. Best time wins.

## Week 1 — Attention foundations

| Movement | Best time | Attempts | Last attempt | Notes |
|----------|-----------|----------|--------------|-------|
| `scaled_dot_product_attention` | 1h43m | 1 | 2026-05-22 | baseline w/ coaching hints; two impls (intermediate-mask + inline `arange + masked_fill`); both match `F.sdpa(is_causal=True)` |
| `multi_head_attention`         | — | 0 | — | |
| `transformer_block`            | — | 0 | — | |
| `positional_encodings`         | — | 0 | — | |

## Warmups & basics

| Movement | Best time | Attempts | Last attempt | Notes |
|----------|-----------|----------|--------------|-------|
| `softmax_stable`  | 20m | 1 | 2026-05-22 | baseline; got per-axis max wrong initially (used global `np.max`), and coupled to torch; both fixed. matches `torch.softmax` on 3D random + 2D mixed-magnitude. |
| `triangular_mask` | — | 0 | — | |
| `cross_entropy`   | — | 0 | — | |
| `layernorm`       | — | 0 | — | |
| `rmsnorm`         | — | 0 | — | |
| `sgd_momentum`    | — | 0 | — | |
| `adam`            | — | 0 | — | |
| `dropout`         | — | 0 | — | |

## LeetCode

| Problem | Category | Best time | Attempts | Last |
|---------|----------|-----------|----------|------|
| Top K Frequent Elements         | arrays | — | 0 | — |
| Product of Array Except Self    | arrays | — | 0 | — |
