# Data Parallel simulated (full)

Multi-process DDP via multiprocessing. Real all-reduce across workers. Verify: matches single-worker training on the full batch, and scales linearly with worker count (throughput measurement).

Total: **90 min - 2 hr.** Prerequisite: `ddp_grad_sync` (numpy sim) done.

## What you'll build

```python
def train_ddp(world_size, model_fn, X, y, epochs, batch_size, lr):
    """
    Spawn world_size workers. Each holds full model. Batch is split.
    After each backward: all-reduce (average) gradients across workers.
    Every worker applies the SAME averaged gradient.
    
    Returns the final trained model (all workers should have identical params).
    """
    ...
```

## Exercises

### Exercise 1 — Basic DDP (45 min)

Implement using `multiprocessing.Process` + `Queue` for the all-reduce:

```python
def worker(rank, world_size, X, y, model_state, epochs, batch_size, lr, all_reduce_channel):
    """
    Each worker:
      - Init identical model (same seed)
      - For each epoch:
        - Take shard of X: X_local = X[rank * B/N : (rank+1) * B/N]
        - Forward
        - Backward → get local grads
        - All-reduce (average) grads via channel
        - Apply averaged grads to local model
      - Return final model (all workers should have identical)
    """
    torch.manual_seed(42)  # SAME seed on every worker → identical init
    W = ...  # init model
    
    per_worker = len(X) // world_size
    
    for epoch in range(epochs):
        X_local = X[rank * per_worker : (rank + 1) * per_worker]
        y_local = y[rank * per_worker : (rank + 1) * per_worker]
        
        # Forward + local grad
        pred = X_local @ W
        loss = ((pred - y_local) ** 2).mean()
        grad = 2 * X_local.T @ (pred - y_local) / len(X_local)
        
        # All-reduce (average across workers)
        grad_avg = all_reduce_average(grad, rank, world_size, all_reduce_channel)
        
        # Update (same for every worker)
        W -= lr * grad_avg
    
    return W
```

### Exercise 2 — Implement all_reduce_average via queues (30 min)

Simple naive version:
- Every worker sends its tensor to worker 0.
- Worker 0 sums, averages, broadcasts back.
- O(N) latency per all-reduce.

Better: ring all-reduce.
- Split tensor into N chunks.
- N-1 rounds of send-right/receive-left: scatter-reduce phase.
- N-1 more rounds: all-gather phase.
- Total per-worker traffic: 2(N-1)/N * tensor_size ≈ 2 * tensor_size.
- (This is the movement `ring_all_reduce_simulated` — reuse if you did that one.)

### Exercise 3 — Verify + measure throughput (30 min)

```python
# Baseline: single-worker training
np.random.seed(0)
X = torch.randn(1000, 8)
true_W = torch.randn(8, 1)
y = X @ true_W + 0.01 * torch.randn(1000, 1)

torch.manual_seed(42)
W_baseline = torch.randn(8, 1) * 0.1
for epoch in range(100):
    pred = X @ W_baseline
    grad = 2 * X.T @ (pred - y) / len(X)
    W_baseline -= 0.01 * grad

# DDP with world_size=4
W_ddp = train_ddp(world_size=4, model_fn=lambda: init_linear(8, 1),
                   X=X, y=y, epochs=100, batch_size=1000, lr=0.01)

# Should be numerically identical (same batch, same order, same seed)
assert torch.allclose(W_ddp, W_baseline, atol=1e-5)

# Also measure throughput
import time
for ws in [1, 2, 4, 8]:
    t0 = time.time()
    train_ddp(world_size=ws, ...)
    elapsed = time.time() - t0
    print(f"world_size={ws}: {elapsed:.2f}s")
# Expect near-linear speedup up to some worker count, then diminishing returns
```

## Watch out for

1. **Identical initialization across workers**: same seed BEFORE the model init.
2. **Batch shard determinism**: worker `r` must get `X[r * B/N : (r+1) * B/N]`, not a random shard. Otherwise gradients differ from single-worker.
3. **All-reduce = AVERAGE, not SUM**: the `k/N` factor matters. Vanilla DDP averages.
4. **Latency vs bandwidth**: for small tensors, naive gather-broadcast may beat ring due to lower overhead. For large tensors (real model grads), ring wins by bandwidth optimality.
5. **What DDP does NOT reduce memory**: every worker still holds the full model, full grads, full optimizer state. That's what ZeRO fixes.
6. **Gradient bucketing**: PyTorch DDP fires the all-reduce as soon as a bucket of gradients is ready, overlapping with backward compute. You don't need to implement this — just know it exists.

## What to say aloud (~2 min)

> "Data parallel replicates the full model on every GPU and splits the batch across GPUs. Each GPU does a full forward + backward on its shard of the batch. Then we all-reduce (average) gradients across all GPUs, and every GPU applies the same averaged gradient. Result: all GPUs stay in sync with identical parameters.
>
> Communication is gradient-sized, per step, per worker. For 70B params fp16, that's 140 GB per all-reduce — even on 400 Gbps InfiniBand, ~3 seconds just for gradient sync. That's why DDP alone doesn't scale beyond a few dozen GPUs for big models.
>
> DDP has one crucial memory limitation: every worker holds the FULL model, gradients, and optimizer state. For Adam mixed precision, that's 16 bytes per param × 70B = 1.12 TB per GPU. Can't fit. This is why FSDP (ZeRO-3) exists — same all-reduce bandwidth, but state is sharded across workers so memory shrinks linearly with world size."

## References

- Li et al. 2020, "PyTorch Distributed: Experiences on Accelerating Data Parallel Training"
- Real implementation: `torch.nn.parallel.DistributedDataParallel`
