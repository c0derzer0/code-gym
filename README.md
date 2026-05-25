# code-gym

Daily coding reps, by hand. To solidify fundamentals in a world of AI coding. Train the muscle, keep it working.

## Why this exists

AI writes most of the code now. Fast, mostly right. The catch: lean on it too much and the underlying muscle gets rusty. Writing attention from scratch. Reasoning through an algorithm cold. Debugging a training loop without help. Reading a paper and turning it into code without a copilot in the loop.

This repo is the gym. Daily ML primitives from scratch and LeetCode mediums, timed. No AI during the timer. References after.

Same movements come back week after week. Form first, speed second. Over time the foundations get solid — fast enough to write what's needed when it's needed, and clear enough on first principles to actually review what the AI produces instead of trusting it blind.

Also useful for interviews where people still test coding unassisted. And it reframes interview prep itself: instead of anxious cramming before a loop, it becomes part of a daily mental workout. Training, sometimes even fun.

## How it works

- **Movements** — atomic from-scratch implementations (`scaled_dot_product_attention`, `adam`, `rmsnorm`, ...). Each lives in `movements/<name>/`, with a problem statement, references, and an `attempts/` folder.
- **WODs (Workouts of the Day)** — daily sessions combining 2–3 movements + a LeetCode "cardio." Logged in `wods/`.
- **PBs (Personal Bests)** — best time-to-correct on each movement. Tracked in `pbs.md` and surfaced in the table below.
- **The Timer Rule** — during a movement's timer (45 min for main, 15 for warmup), no AI, no docs, no Stack Overflow. References come *after*. This is the whole point.

## Daily structure

| Day type | Format |
|----------|--------|
| ML day (×4/week) | 45 min main movement + 15 min warmup movement |
| LC day (×2/week) | 30 min LeetCode medium + 30 min basic deep-dive |
| Rest (×1/week) | Sunday retro + plan next week |

## Personal bests

| Movement | Best time | Attempts | Last |
|----------|-----------|----------|------|
| `scaled_dot_product_attention` | 1h43m  | 1 | 2026-05-22 |
| `multi_head_attention`         | ~1h40m | 1 | 2026-05-24 |
| `adam`                         | ~1h15m | 1 | 2026-05-25 |
| `softmax_stable`               | 20m    | 1 | 2026-05-22 |
| `triangular_mask`              | 11m    | 1 | 2026-05-23 |

Full table in [`pbs.md`](pbs.md). LeetCode times tracked there too.

## Recent WODs

- [Day 3 — 2026-05-25](wods/day03.md) — LC day: Top K Frequent Elements (two impls) + Adam deep-dive (matches `torch.optim.Adam` to 1e-5).
- [Day 2 — 2026-05-23](wods/day02.md) — Attention foundations: MHA (main) + triangular mask (warmup). Causal property verified by perturbation test.
- [Day 1 — 2026-05-22](wods/day01.md) — Attention foundations: SDPA (main) + stable softmax (warmup). Both verified vs PyTorch reference.

## Repo map

```
movements/    one folder per atomic implementation
wods/         daily workout logs
leetcode/     LC mediums by category
PLAN.md       rotation backbone (6-week cycle)
pbs.md        source of truth for PB table
```

## Setup

Managed with [uv](https://docs.astral.sh/uv/). After cloning:

```bash
uv sync                                              # install deps from uv.lock
uv run python movements/scaled_dot_product_attention/attempts/attempt_1.py
uv run pytest                                        # if tests are wired in
uv run jupyter lab                                   # for notebook exploration
```

Python 3.11+, PyTorch with MPS support on macOS.
