# Day 2 — 2026-05-23

**Theme:** Attention foundations — Day 2 (close-out spanned into 2026-05-24)

## Warmup (15 min target): `triangular_mask`

- Started: 11:00am
- Finished: 11:11am (~11 min)
- **PB set:** 11m baseline. Beats the 15-min warmup target.
- Two variants implemented:
  - `causal_mask(seq_len)`: bool tensor via `arange[:, None] <= arange[None, :]`. True = allowed (j ≤ i).
  - `causal_mask_additive(seq_len)`: 0/-inf tensor via `masked_fill(~mask, -inf)`.
- One round of feedback caught: the bool convention was initially inverted (True where masked). Flipped to match the README spec (True = allowed). Additive function then needed `~mask` before `masked_fill` to stay correct — design choice, not accident.
- Both outputs match the README spec exactly.
- Followups for attempt 2:
  - Typo `mask_addtive` → `mask_additive`.
  - Optional: build additive directly from `arange > arange` and skip the bool detour.

## Main (45 min target): `multi_head_attention` (+ causal mask)

- Started: 11:21am
- First pass (runs without error): 12:57pm (~1h36m)
- Correct (shape preserved + causality property holds): close-out on 2026-05-24
- **PB set:** ~2h baseline (with multiple rounds of hints; spanned two sessions). Next attempt target: <45 min unaided.
- Architecture:
  - Fused QKV projection `nn.Linear(d_model, 3*d_model)`.
  - Reshape to `(B, T, n_heads, 3*d_head)`, permute to `(B, n_heads, T, 3*d_head)`, then `qkv.chunk(3, dim=-1)` for Q/K/V.
  - Attention with causal mask added to logits before softmax.
  - Output projection `nn.Linear(d_model, d_model)` at the end.
  - Causal mask registered as a buffer (so it moves with `.to(device)`).
- Verification: shape assertion `out.shape == x.shape`; causality property — perturb `x[:, t, :]`, assert `out[:, :t, :]` unchanged. Both pass.
- Why not `allclose` vs `torch.nn.MultiheadAttention`? Different output projection conventions and random inits — without copying weights, it'd never match. Structural tests (shape + causality) are the right proof here.
- Bugs caught + fixes (detailed in the file header):
  - `causal_mask_additive` missing `self` — hidden until `causal=True`.
  - Causal mask logic was inverted (was masking the past). Flipped `col < row` → `col > row`.
  - `register_buffer('casual_mask', ...)` had a typo AND was duplicating state with a pre-assigned `self.causal_mask`. Fix: register once, no separate assignment.
  - Missing output projection — heads couldn't mix.
  - `view` after `permute` (non-contiguous); switched to `reshape`.
  - Q/K/V split was scattered across two methods; consolidated via `qkv.chunk(3, dim=-1)` in `forward`.
- Followups for attempt 2:
  - Constructor takes `seq_len`; standard pattern is `max_seq_len` + slice `[:T, :T]` in forward so the module handles variable sequence lengths.
  - Strip debug `print`s left in for visibility.
  - Header line says "Attempt N" — replaced with "Attempt 1".

## Retro (one line)

The bugs that hid behind `causal=False` (missing `self`, inverted mask) are why you turn on the harder code path early — visual inspection of a passing test means nothing if the test doesn't exercise the bug.
