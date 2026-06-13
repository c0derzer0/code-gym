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

## Week 2 — LLM inference core

| Day | Main (45 min) | Warmup / LC deep-dive |
|-----|---------------|-----------------------|
| 8   | `kv_cache` (extend Week 1's MHA)         | `cross_entropy` |
| 9   | `sampling_greedy_temperature`            | `embedding_lookup` |
| 10  | LC: Course Schedule (topo sort)          | Deep-dive: `sgd_momentum` |
| 11  | `sampling_top_k_top_p`                   | `dataset_dataloader` |
| 12  | `batched_inference_padding`              | `training_loop_skeleton` |
| 13  | LC: Clone Graph                          | Deep-dive: `residual_block` |
| 14  | Rest + retro                             | — |

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
