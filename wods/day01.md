# Day 1 — 2026-05-22

**Theme:** Attention foundations — Day 1

## Main (45 min target): `scaled_dot_product_attention`

- Started: 7:17pm
- First pass (runs without error): 8:10pm (~53 min)
- Correct (passes `allclose` vs `F.scaled_dot_product_attention(is_causal=True)`): ~9:00pm
- Total wall: ~1h43m (with coaching hints)
- **PB set:** 1h43m baseline. Next attempt target: <30 min unaided.
- Two implementations:
  - `attention`: built an intermediate (0, -inf) mask tensor via `triu` + `masked_fill`, then `softmax(attn + mask)`.
  - `attention_mask_optimized`: inline `arange[None,:] > arange[:,None]` directly inside `masked_fill` on `attn` — no intermediate mask tensor.
- Stuck on (and lessons):
  - Initial bug: applied mask multiplicatively (`attn * mask`) — breaks for negative scores. **Mask is additive.**
  - Second bug: mask values were `(1, -inf)` not `(0, -inf)`. Output was still correct via softmax translation invariance, but unconventional.
  - Resolved by switching mask construction to start from zeros (or equivalently using `masked_fill` directly on `attn`).
- Followups (to address in attempt 2):
  - Typo `casual_mask` → `causal_mask`.
  - Signature: take `mask=None` tensor instead of boolean `use_mask` — more flexible (padding masks, sliding windows).
  - Cache the causal-mask tensor outside the function (registered buffer, sliced per call) instead of rebuilding every forward.

## Warmup (15 min target): `softmax_stable`

- Started: 10:20pm
- First pass (runs without error): 10:27pm (~7 min)
- Correct (passes `allclose` vs `torch.softmax`): 10:40pm
- Total wall: ~20 min
- **PB set:** 20m baseline. Next attempt target: <8 min unaided, clean.
- Stuck on (and lessons):
  - First pass used `np.max(x)` (global) instead of `np.max(x, axis, keepdims=True)` — works for 1D but breaks for 2D when magnitudes differ across rows. The stability point is *per-axis* max, not global.
  - First pass also coupled the function to torch (`x.numpy()` in, `torch.tensor(out)` out). Function should be pure numpy; framework conversion belongs in the test block.
- Cleanup in this attempt: renamed function `softmax` → `compute_softmax`, local `sum` → `axis_sum` (no more builtin/name shadows).
- Followups (to address in attempt 2):
  - Use explicit `dim=-1` in test comparisons rather than relying on `axis == -1` by coincidence for 2D inputs.

## Retro (one line)

Multiplicative-vs-additive mask is the kind of mistake you only make once; the iteration loop with the coach was worth more than the time it cost.
