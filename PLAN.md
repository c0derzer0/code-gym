# Plan

The rotation. Six-week backbone. Each Sunday: review, re-prioritize next week's row if signal warrants.

## Day cadence (whichever calendar day)

- **Day 1, 2, 4, 5** — ML days: 45 min main movement + 15 min warmup
- **Day 3, 6** — LC days: 30 min LC medium + 30 min basic deep-dive
- **Day 7** — Rest + Sunday retro

If extra time on a day, do another attempt of a past movement and try to set a new PB.

## Week 1 — Attention foundations

| Day | Main (45 min) | Warmup / LC deep-dive (15 / 30 min) |
|-----|---------------|-------------------------------------|
| 1   | `scaled_dot_product_attention`         | `softmax_stable` |
| 2   | `multi_head_attention` (+ causal mask) | `triangular_mask` |
| 3   | LC: Top K Frequent Elements            | Deep-dive: `adam` |
| 4   | `transformer_block` (MHA+FFN+res+LN)   | `layernorm` |
| 5   | `positional_encodings` (sinusoidal + RoPE) | `rmsnorm` |
| 6   | LC: Product of Array Except Self       | Deep-dive: `dropout` |
| 7   | Rest + retro                            | — |

Note: `sgd_momentum` rolls into Week 2's warmup rotation. Order rationale: `triangular_mask` lands on Day 2 because it's the primitive MHA needs that same session; `layernorm` lands on Day 4 because the transformer block uses it.

## Week 2 — LLM inference core

| Day | Main (45 min) | Warmup / LC deep-dive |
|-----|---------------|-----------------------|
| 8   | `kv_cache` (extend Week 1's MHA)         | `cross_entropy` |
| 9   | `sampling_greedy_temperature`            | `embedding_lookup` |
| 10  | LC: Longest Substring Without Repeating  | Deep-dive: `sgd_momentum` |
| 11  | `sampling_top_k_top_p`                   | `dataset_dataloader` |
| 12  | `batched_inference_padding`              | `training_loop_skeleton` |
| 13  | LC: 3Sum                                 | Deep-dive: `residual_block` |
| 14  | Rest + retro                             | — |

## Week 3 — Inference advanced (algo + memory, interleaved)

| Day | Main (45 min) | Warmup / LC deep-dive |
|-----|---------------|-----------------------|
| 15  | `paged_kv_cache` (memory: block-table KV)    | `block_indexing` (paging primitive) |
| 16  | `grouped_query_attention` (algo: Llama-3 style) | `repeat_interleave_kv` (GQA primitive) |
| 17  | LC: Number of Islands (LC 200)               | Deep-dive: re-attempt `scaled_dot_product_attention` (speed PB) |
| 18  | `quantize_linear_int8` (memory: weight-only) | `int8_quantize_dequantize_pair` |
| 19  | `sliding_window_attention` (algo: Mistral)   | Re-attempt: `triangular_mask` as band-mask variant |
| 20  | LC: Course Schedule (LC 207)                 | Deep-dive: `flash_attention_simplified` (online-softmax recurrence) |
| 21  | Rest + retro                                 | — |

Week 3 mixes 2 memory mains + 2 algo mains. Pairings: paged_kv_cache pairs with the GQA day (KV cache memory savings stack); quantization sits independently; sliding window naturally pairs with bounded KV cache. Dropped: beam_search (legacy), speculative_decoding (push to W4/W6), continuous batching (scheduler — discuss as system design, not a kata).

## Future weeks (6-week backbone — adjust based on interview signal)

| Week | Theme |
|------|-------|
| 1 | Attention foundations |
| 2 | LLM inference core |
| 3 | Inference advanced — algo + memory mixed (paged KV, GQA, INT8 quant, sliding window) |
| 4 | Fine-tuning (LoRA, SFT, reward model, DPO) — also a likely landing spot for speculative_decoding + flash_attention full impl |
| 5 | RLHF + mini-GPT capstone |
| 6 | Synthesis — adaptive, weakest movements under timer, slot for whatever interview signal demands |

Each Sunday: write retro in the current week's WOD log, update next week's row. Plan is intentionally not locked beyond Week 3 — re-prioritize as interviews come in.

## Sunday retro template

1. PBs set this week?
2. Movements that felt slow — schedule a re-attempt next week.
3. Any interview signal (real or upcoming) that should shift next week's theme?
