# Personal bests

Best **time-to-correct** per movement. "Correct" = passes the test block in the movement's README.

Update after each attempt. Best time wins.

## Week 1 — Attention foundations

| Movement | Best time | Attempts | Last attempt | Notes |
|----------|-----------|----------|--------------|-------|
| `scaled_dot_product_attention` | 1h43m | 1 | 2026-05-22 | baseline w/ coaching hints; two impls (intermediate-mask + inline `arange + masked_fill`); both match `F.sdpa(is_causal=True)` |
| `multi_head_attention`         | ~2h | 1 | 2026-05-24 | baseline w/ hints across two sessions; fused QKV + chunk split + causal-mask buffer + out_proj; passes shape + causality property tests |
| `transformer_block`            | ~43m | 1 | 2026-05-30 | baseline w/ hints; pre-norm MHA + FFN + 2× LayerNorm + residuals; reuses MHA + LayerNorm from earlier days; shape preserves + grad flows + handles T < max_seq_len (forced a post-hoc fix in MHA to slice the mask) |
| `kv_cache`                     | ~1 day | 1 | 2026-07-07 | Extend MHA with `forward(x, cache=None)` supporting incremental generation. Verified: full-sequence forward matches one-token-at-a-time cached generation to 1e-5. Baseline w/ several hint rounds — bugs: causal mask returned bool tensor (bool + attn added 1.0 instead of -inf), permute with shape values instead of axis indices, nn.linear typo, cache=None handling in the write path. |
| `online_softmax_recurrence`    | — | 0 | — | Flash Attention's core algorithm. Numpy. Two-pass: pass 1 computes running (m, d) chunk-by-chunk with rescale on max updates; pass 2 computes output. Verify allclose vs torch.softmax to 1e-6. |
| `grouped_query_attention`      | — | 0 | — | MHA variant with n_kv_heads < n_q_heads. K/V broadcast across Q groups via repeat_interleave. Llama-3 style. Verify shape + causality property. |
| `positional_encodings`         | ~73m | 1 | 2026-06-04 | baseline w/ hints; sinusoidal w/ sin+phase trick + RoPE w/ clever mask-and-stack companion vector. RoPE verified via relative-position invariance (inner product depends only on m-n). Standard half-d / Llama rotate-half ~2× more efficient — followup. |

## Foundational

| Movement | Best time | Attempts | Last attempt | Notes |
|----------|-----------|----------|--------------|-------|
| `backprop_from_scratch` | — | 0 | — | 2-layer MLP binary classifier in pure numpy. Forward + backward + training loop. Convergence: accuracy > 90% on toy task. Re-rep periodically — the substrate every ML training loop is built on. |
| `ring_all_reduce_simulated` | — | 0 | — | N processes via `multiprocessing`, ring algorithm, scatter-reduce + all-gather phases. Bandwidth-optimal all-reduce, the foundation of NCCL. Test: all ranks match `sum(locals)` + per-rank comm ~ `2(N-1)/N · B`. |

## Warmups & basics

| Movement | Best time | Attempts | Last attempt | Notes |
|----------|-----------|----------|--------------|-------|
| `softmax_stable`  | 20m | 1 | 2026-05-22 | baseline; got per-axis max wrong initially (used global `np.max`), and coupled to torch; both fixed. matches `torch.softmax` on 3D random + 2D mixed-magnitude. |
| `triangular_mask` | 11m | 1 | 2026-05-23 | beats 15-min warmup target; bool (True = allowed, j ≤ i) + additive (0/-inf) variants from arange comparison |
| `cross_entropy`   | — | 0 | — | |
| `layernorm`       | ~40m | 1 | 2026-05-26 | baseline w/ hints; nn.Parameter γ/β (not buffers), biased variance via (x-mean)^2 mean (not torch.var), matches F.layer_norm within 1e-5 |
| `rmsnorm`         | — | 0 | — | |
| `sgd_momentum`    | — | 0 | — | |
| `adam`            | ~1h15m | 1 | 2026-05-25 | baseline w/ hints; matches `torch.optim.Adam` over 50 steps to 1e-5; bias correction via m_hat/v_hat temps, not stored back into state |
| `dropout`         | — | 0 | — | |

## LeetCode

| Problem | Category | Best time | Attempts | Last |
|---------|----------|-----------|----------|------|
| Top K Frequent Elements         | arrays | ~1h | 1 | 2026-05-25 | baseline w/ hints; two impls (incremental hash-of-hashes + canonical bucket sort); both pass 5 tests incl. edge cases |
| Product of Array Except Self    | arrays | — | 0 | — |
