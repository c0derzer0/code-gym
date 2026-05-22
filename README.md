# code-gym

A daily training repo for ML coding reflexes. Built like a gym, not a course.

## What this is

Most "interview prep" repos are short-lived sprints toward a single loop. This isn't that. It's a continuous training log — atomic from-scratch implementations of ML primitives (attention, layernorm, optimizers, samplers, RL losses) done under a timer, then compared against canonical references. Each implementation is a **movement**. The first attempt sets a baseline; subsequent attempts try to beat it.

Think powerlifting log meets coding kata: same movements come back periodically, times get shorter, and the gap between "I know this conceptually" and "I can write it cold in 6 minutes" closes.

## How it works

- **Movements** — atomic from-scratch implementations (`scaled_dot_product_attention`, `adam`, `rmsnorm`, ...). Each lives in `movements/<name>/`, with a problem statement, references, and an `attempts/` folder.
- **WODs (Workouts of the Day)** — daily sessions combining 2–3 movements + a LeetCode "cardio." Logged in `wods/`.
- **PRs (Personal Records)** — best time-to-correct on each movement. Tracked in `prs.md` and surfaced in the table below.
- **The Timer Rule** — during a movement's timer (45 min for main, 15 for warmup), no AI, no docs, no Stack Overflow. References come *after*. This is the whole point.

## Daily structure

| Day type | Format |
|----------|--------|
| ML day (×4/week) | 45 min main movement + 15 min warmup movement |
| LC day (×2/week) | 30 min LeetCode medium + 30 min basic deep-dive |
| Rest (×1/week) | Sunday retro + plan next week |

## Personal records

| Movement | Best time | Attempts | Last |
|----------|-----------|----------|------|
| _will populate after Day 1_ | — | — | — |

## Recent WODs

_will populate after Day 1_

## Repo map

```
movements/    one folder per atomic implementation
wods/         daily workout logs
leetcode/     LC mediums by category
PLAN.md       rotation backbone (6-week cycle)
prs.md        source of truth for PR table
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
