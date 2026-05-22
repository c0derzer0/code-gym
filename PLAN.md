# Plan

The rotation. Six-week backbone. Each Sunday: review, re-prioritize next week's row if signal warrants.

## Day cadence (whichever calendar day)

- **Day 1, 2, 4, 5** — ML days: 45 min main movement + 15 min warmup
- **Day 3, 6** — LC days: 30 min LC medium + 30 min basic deep-dive
- **Day 7** — Rest + Sunday retro

If extra time on a day, do another attempt of a past movement and try to PR.

## Week 1 — Attention foundations

| Day | Main (45 min) | Warmup / LC deep-dive (15 / 30 min) |
|-----|---------------|-------------------------------------|
| 1   | `scaled_dot_product_attention`         | `softmax_stable` |
| 2   | `multi_head_attention` (+ causal mask) | `layernorm` |
| 3   | LC: Top K Frequent Elements            | Deep-dive: `adam` |
| 4   | `transformer_block` (MHA+FFN+res+LN)   | `rmsnorm` |
| 5   | `positional_encodings` (sinusoidal + RoPE) | `sgd_momentum` |
| 6   | LC: Product of Array Except Self       | Deep-dive: `dropout` |
| 7   | Rest + retro                            | — |

## Week 2 — LLM inference core (preview)

KV cache · Greedy + temperature sampling · Top-k + top-p · Batched inference w/ padding
Warmups continue rotation: `cross_entropy`, `embedding_lookup`, `dataset_dataloader`, `training_loop_skeleton`

## Future weeks (6-week backbone)

| Week | Theme |
|------|-------|
| 1 | Attention foundations |
| 2 | LLM inference core |
| 3 | Inference advanced (beam, GQA, sliding window, speculative) |
| 4 | Fine-tuning (LoRA, SFT, reward model, DPO) |
| 5 | RLHF + mini-GPT capstone |
| 6 | Synthesis — adaptive, weakest movements under timer |

Each Sunday: write retro in the current week's WOD log, update next week's row.

## Sunday retro template

1. PRs set this week?
2. Movements that felt slow — schedule a re-attempt next week.
3. Any interview signal (real or upcoming) that should shift next week's theme?
