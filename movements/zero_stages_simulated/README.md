# ZeRO stages simulated (full)

Implement all three ZeRO stages via `multiprocessing`. Verify all workers converge to the same params. Measure memory + communication per stage.

Total: **2-3 hr.** Prerequisite: `ddp_grad_sync` done + `zero_memory_accounting` understood.

## What you'll build

An `nn.Module`-esque interface where you can specify a ZeRO stage and run distributed training on a toy MLP or transformer block. All in one process with multiprocessing spawning "workers."

```python
def train_zero(model_fn, world_size, stage, X, y, epochs, lr):
    """
    Spawn world_size worker processes, each running training with the
    specified ZeRO stage.
    
    Stages:
      0: DDP baseline (params replicated, grads all-reduced, opt replicated)
      1: Optimizer state sharded (12 bytes/param → 12/N per worker)
      2: + Gradients sharded (reduce-scatter instead of all-reduce)
      3: + Params sharded (all-gather per layer during forward)
    
    Verify: after training, all workers hold identical (or reassembled) params
    and match the DDP baseline's final loss.
    """
    ...
```

## Exercise 1 — ZeRO-1 (30 min)

Shard optimizer state. Each worker owns a slice of the fp32 master params + Adam m + v.

Pattern:
- Forward + backward as DDP (all-reduce gradients).
- Optimizer step: each worker updates ONLY its shard of the master params.
- After step: **all-gather** the updated params → everyone has the fresh fp16 params.

Communication vs DDP: same all-reduce for grads, PLUS all-gather for params after step. Slightly more.

## Exercise 2 — ZeRO-2 (45 min)

Now shard gradients too. Each worker only holds its shard.

Pattern:
- Forward as before.
- Backward locally, then **reduce-scatter** gradients (each worker gets the reduced version of its shard, not the full averaged grad).
- Optimizer step: each worker updates its shard using its shard's grad.
- All-gather updated params.

Communication vs ZeRO-1: reduce-scatter (=N/N all-reduce) + all-gather. Same total bandwidth as DDP all-reduce.

## Exercise 3 — ZeRO-3 (60 min)

Now shard params too. Each worker only holds its shard of params, grads, and opt state.

Pattern (per layer during forward):
- Worker needs params — **all-gather** them.
- Forward through the layer using the gathered params.
- **Free** the gathered params (only keep own shard).
- Repeat for each layer.

Backward:
- Similar all-gather-then-free pattern for params in reverse.
- Reduce-scatter grads at layer's end.

Optimizer step: each worker updates only its shard.

Communication vs ZeRO-2: one extra all-gather per layer per forward, and one per layer per backward. Roughly 1.5× DDP bandwidth.

## Test

```python
np.random.seed(0)
N, D = 200, 8
X = np.random.randn(N, D)
true_W = np.random.randn(D, 1)
y = X @ true_W + 0.01 * np.random.randn(N, 1)

# Train DDP baseline
baseline_params = train_zero(model_fn=lambda: init_mlp(D, hidden=16),
                              world_size=4, stage=0, X=X, y=y, epochs=100, lr=0.01)

# Train ZeRO-1/2/3
for stage in [1, 2, 3]:
    zero_params = train_zero(model_fn=lambda: init_mlp(D, hidden=16),
                              world_size=4, stage=stage, X=X, y=y, epochs=100, lr=0.01)
    # After training, worker 0's assembled params should match the baseline
    assembled = assemble_params(zero_params)  # gather shards → full model
    assert np.allclose(assembled, baseline_params, atol=1e-4), f"ZeRO-{stage} diverged"
    print(f"✓ ZeRO-{stage} matches DDP baseline")
```

## Watch out for

1. **Determinism across workers**: use the same `torch.manual_seed` in every worker, and the same batch shuffling. Otherwise gradients will differ and you can't verify equivalence.
2. **All-gather memory spike in ZeRO-3**: during a layer's forward, ALL workers momentarily hold the full param tensor. Real FSDP frees after each layer to keep memory scaling linear.
3. **`torch.multiprocessing.spawn`** vs `multiprocessing.Process`: for PyTorch, use `torch.multiprocessing`.
4. **Communication primitives**: implement your own tiny `all_reduce(tensor, group)`, `reduce_scatter(tensor, group)`, `all_gather(tensor, group)` using pipes or shared memory. These are the operations NCCL provides in real distributed setup.
5. **Backward through frozen params in ZeRO-3**: gradients accumulate only on your own shard.

## What to say aloud (~2 min)

> "ZeRO shards different parts of the training state across workers. ZeRO-1 shards optimizer state — the biggest single component at 12 bytes per param — while replicating params and gradients. Communication is standard DDP all-reduce for grads plus a per-step all-gather to broadcast updated params.
>
> ZeRO-2 additionally shards gradients: instead of all-reducing full gradients, we reduce-scatter, so each worker only receives its shard's averaged gradient. Same total bandwidth as DDP.
>
> ZeRO-3 shards everything — params, grads, optimizer state. Each worker only stores its shard of each. During forward, we all-gather the params for the current layer, compute, then free. Reverse pattern in backward. This gives linear memory scaling with world size, at the cost of ~1.5× DDP communication bandwidth.
>
> ZeRO-3 (equivalent to PyTorch FSDP) is what makes training frontier models possible. Llama-3-70B trained on 24k GPUs would be impossible with DDP — you literally cannot fit even one copy of the model on a single GPU. ZeRO-3 shards it into pieces small enough to fit."

## References

- Rajbhandari et al. 2020, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- Zhao et al. 2023, "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"
- Real implementation: `torch.distributed.fsdp`
