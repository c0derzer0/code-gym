# ZeRO memory accounting

The memory math for ZeRO-1/2/3. Not a simulation — a calculator + reasoning exercise. This is the "explain the memory savings of FSDP" answer.

Total: ~30 min.

## The setup

For a model with `P` parameters trained with Adam in mixed precision:

| Component | Per-param bytes | Notes |
|---|---|---|
| Model params (fp16) | 2 | For forward |
| Gradients (fp16) | 2 | For backward |
| Optimizer state (fp32 master params) | 4 | Adam m + v + fp32 master → **12 bytes/param actually** (m: 4, v: 4, master: 4) |
| Adam m (fp32) | 4 | First moment |
| Adam v (fp32) | 4 | Second moment |
| **Total per param** | **16 bytes** | 2 + 2 + 4 + 4 + 4 |

Wait — 16 bytes/param for Adam mixed precision. **For a 70B model: 70 · 16 = 1.12 TB.** Doesn't fit on one GPU. That's the problem ZeRO solves.

## ZeRO stages

| Stage | Sharded | Replicated | Per-GPU bytes/param (world=N) |
|---|---|---|---|
| ZeRO-0 (DDP) | — | params, grads, opt | 16 |
| ZeRO-1 | opt (12) | params, grads | 4 + 12/N |
| ZeRO-2 | opt, grads | params | 2 + 2/N + 12/N |
| ZeRO-3 (FSDP) | opt, grads, params | — | 2/N + 2/N + 12/N = 16/N |

**ZeRO-3 gives linear memory scaling with N.** For 70B on 32 GPUs: 1.12 TB / 32 = 35 GB per GPU. Fits.

## Exercise 1 — Calculator (15 min)

```python
def zero_memory_per_gpu(param_count: int, world_size: int, stage: int, dtype_bytes: int = 2) -> dict:
    """
    Return per-GPU memory breakdown in bytes for the given ZeRO stage.
    
    Assumes mixed precision Adam:
      - params: dtype_bytes (fp16 = 2) per param
      - grads: dtype_bytes per param
      - opt state: 12 bytes per param (fp32 master + m + v)
    
    Returns {'params': int, 'grads': int, 'opt_state': int, 'total': int}.
    """
    ...
```

Test:
```python
r = zero_memory_per_gpu(70_000_000_000, world_size=32, stage=3)
print(f"ZeRO-3 70B on 32 GPUs: {r['total'] / 1e9:.1f} GB per GPU")
# expected: ~35 GB (from 16/N formula)
```

## Exercise 2 — Trade-off table (15 min)

Fill in this table for a 70B model:

| Stage | 8 GPUs | 32 GPUs | 128 GPUs | 512 GPUs |
|---|---|---|---|---|
| ZeRO-0 | 1120 GB | 1120 GB | 1120 GB | 1120 GB |
| ZeRO-1 | 388 GB | 315 GB | 291 GB | 283 GB |
| ZeRO-2 | 140 GB | 44 GB | 12 GB | 3 GB |
| ZeRO-3 | 140 GB | 35 GB | 8.75 GB | 2.19 GB |

(Compute these with your function; check them against the formulas.)

Communication cost:
- **ZeRO-1**: same as DDP (all-reduce grads).
- **ZeRO-2**: same total bandwidth as DDP, but reshuffled.
- **ZeRO-3**: **1.5× DDP bandwidth**. All-gather params per layer (forward + backward) + reduce-scatter grads.

The trade: **ZeRO-3 gets memory scaling in exchange for 1.5× communication.**

## What to say aloud

> "Mixed-precision Adam training needs 16 bytes per parameter total: 2 for fp16 params, 2 for fp16 gradients, and 12 for fp32 optimizer state (master params + Adam's m and v). For a 70B model, that's 1.12 TB, which obviously doesn't fit on one GPU.
>
> ZeRO fixes this by sharding pieces across the world. ZeRO-1 shards optimizer state — biggest single win, ~4× reduction. ZeRO-2 also shards gradients. ZeRO-3 (equivalent to PyTorch FSDP) shards everything including params, giving linear memory scaling — 70B on 32 GPUs is 35 GB per GPU, which fits.
>
> Cost: ZeRO-3 all-gathers params per layer during forward and backward, then reduce-scatters grads. About 1.5× the communication of vanilla DDP. Worth it because without ZeRO-3 you can't train the model at all."

## References

- Rajbhandari et al. 2020, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- PyTorch FSDP paper (Zhao et al. 2023)
