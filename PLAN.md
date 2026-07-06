# Plan

The next 1-2 weeks of concrete movements. Anything beyond that is decided at Sunday retro by browsing [`CATALOG.md`](CATALOG.md).

## Daily cadence (whichever calendar day)

- **Day 1, 2, 4, 5** — ML days: 45 min main movement + 15 min warmup
- **Day 3, 6** — LC days: 30 min LC medium + 30 min basic deep-dive
- **Day 7** — Rest + Sunday retro

If extra time on a day, do another attempt of a past movement and try to set a new PB.

**One movement at a time. Rotate categories across the week. Don't try to do the whole catalog — that's years of work, not weeks.**

## Week 1 — Attention foundations (in progress)

| Day | Main (45 min) | Warmup / LC deep-dive |
|-----|---------------|-----------------------|
| 1   | `scaled_dot_product_attention` ✓             | `softmax_stable` ✓ |
| 2   | `multi_head_attention` ✓                     | `triangular_mask` ✓ |
| 3   | LC: Top K Frequent ✓                          | Deep-dive: `adam` ✓ |
| 4   | `transformer_block` ✓                        | `layernorm` ✓ |
| 5   | `positional_encodings` ✓                     | `rmsnorm` (pending) |
| 6   | LC: Number of Islands                        | Deep-dive: `dropout` |
| 7   | Rest + retro                                 | — |

## Week 2 — LLM inference optimizations (tilted for a Friday morning coding session)

| Day | Main (~90 min main + explain aloud) | Notes |
|-----|-------------------------------------|-------|
| 8 (Mon)  | `kv_cache` — extend Week 1's MHA with `forward(x, cache=None)` | Verify: full-seq vs one-token-at-a-time cached generation match to 1e-5 |
| 9 (Tue)  | Morning: `online_softmax_recurrence` — Flash Attention's core algorithm (pure numpy). Afternoon: `grouped_query_attention` — MHA variant with `n_kv_heads < n_q_heads` | Verify online softmax matches `torch.softmax`; GQA verifies same output shape + causality property |
| 10 (Wed) | `quantize_linear_int8` — symmetric per-tensor weight quantization | Verify SNR ~40 dB vs fp32 linear |
| 11 (Thu) | End-to-end mock: implement MHA-with-KV-cache from scratch under 45 min timer. Explain FA memory savings aloud. **Rest evening.** | No new code — reinforcement |
| 12 (Fri) | Rest day (morning is off-repo) | — |
| 13 (Sat) | Back to regular Week 2 plan: `sampling_greedy_temperature` + LC | — |
| 14 (Sun) | `sampling_top_k_top_p` + Sunday retro | — |

Held over (was in Week 2, moved to Week 3): `batched_inference_padding`, `dataset_dataloader`, `training_loop_skeleton`, `embedding_lookup`, `cross_entropy`, LC Clone Graph, LC Course Schedule.

## Beyond Week 2 — Sunday-retro driven

Don't pre-plan further. Each Sunday:

1. **Browse [`CATALOG.md`](CATALOG.md)** for next week's 4 main movements + 2 LC mediums.
2. **Mix tracks.** Pull from different categories (math, kernels, systems, ML core, etc.) so the week stays varied.
3. **Use weakness signal.** If a past movement felt slow, schedule a re-attempt as next week's warmup.
4. **One movement / day, rotate categories.** That's the rule.

## Bonus reps available (when you have extra time)

- `leetcode/graphs/pacific_atlantic_water_flow/` — multi-source DFS
- `leetcode/graphs/rotting_oranges/` — multi-source BFS
- `leetcode/graphs/word_ladder/` — BFS + pattern bucketing
- `leetcode/trees/lowest_common_ancestor/` — recursive LCA
- `movements/backprop_from_scratch/` — 2-layer MLP, pure numpy. Re-rep periodically; the substrate every ML training loop is built on.
- `movements/ring_all_reduce_simulated/` — bandwidth-optimal collective via multiprocessing. Foundation of NCCL.

## Sunday retro template

Three sentences. No more.

1. PBs set this week?
2. Movements that felt slow — re-attempt next week.
3. Next week: 4 mains + 2 LCs picked from CATALOG.md. Which tracks?
