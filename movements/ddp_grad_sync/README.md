# DDP grad sync — simulated (small)

Muscle-memory version of data-parallel training. Simulate N workers with independent models on shards of the batch. Sum gradients across workers before the step. Verify: equivalent to single-worker training on the full batch.

Total: ~30 min.

## The pattern

- Each worker holds a **full copy** of the model.
- Batch is **split** across workers: worker r gets `X[r * B/N : (r+1) * B/N]`.
- Forward + backward locally.
- **All-reduce (average) gradients** across workers.
- All workers do `optimizer.step()` with the same averaged gradient → all models stay in sync.

Contrast with TP:
- **DP**: model replicated, batch split, comm = **gradients** (backward only)
- **TP**: model split, batch replicated, comm = **activations** (forward + backward)

## Exercise 1 — Implement (20 min)

```python
def train_ddp_simulated(X, y, world_size, epochs, lr):
    """
    Simulate `world_size` workers doing DDP.
    - Each worker holds its own model (initialized identically).
    - Each epoch:
        - Split batch across workers.
        - Each worker computes grads locally.
        - Average grads across workers (all-reduce simulation).
        - Each worker applies the SAME averaged grad.
    - After training, all workers' models should be identical.
    """
    # Initialize N identical models
    np.random.seed(0)
    W_init = np.random.randn(X.shape[1], 1) * 0.1
    models = [W_init.copy() for _ in range(world_size)]
    
    N, D = X.shape
    per_worker = N // world_size
    
    for epoch in range(epochs):
        grads_per_worker = []
        for rank in range(world_size):
            # Local shard
            X_local = X[rank * per_worker : (rank + 1) * per_worker]
            y_local = y[rank * per_worker : (rank + 1) * per_worker]
            
            # Local forward + backward (simple linear regression: MSE loss)
            pred = X_local @ models[rank]
            loss_local = ((pred - y_local) ** 2).mean()
            grad_local = 2 * X_local.T @ (pred - y_local) / len(X_local)
            grads_per_worker.append(grad_local)
        
        # All-reduce (average)
        avg_grad = sum(grads_per_worker) / world_size
        
        # Every worker applies the SAME averaged grad
        for rank in range(world_size):
            models[rank] -= lr * avg_grad
    
    return models
```

## Test

```python
np.random.seed(0)
N, D = 200, 4
X = np.random.randn(N, D)
true_W = np.random.randn(D, 1)
y = X @ true_W + 0.01 * np.random.randn(N, 1)

# Train with DDP simulation
models = train_ddp_simulated(X, y, world_size=4, epochs=100, lr=0.05)

# Verify: all workers converged to the same model
for rank in range(1, 4):
    assert np.allclose(models[rank], models[0], atol=1e-9)

# Verify: converged near true_W
assert np.allclose(models[0], true_W, atol=0.1)
print(f"✓ all 4 workers identical; W diff from truth: {np.abs(models[0] - true_W).max():.4f}")
```

## Exercise 2 — Comm analysis (10 min)

Data parallel does **one all-reduce of gradients per step**. Gradient size = parameter count.

For Llama-70B: ~70 · 10^9 params, fp16 (assuming grad-in-fp16 accumulation, which is aggressive). All-reduce ≈ 140 GB. Even on 400 Gbps IB, that's ~3 seconds per step just for gradient sync.

**This is why FSDP/ZeRO-3 wins over vanilla DDP for big models: instead of all-reducing full gradients, it reduce-scatters them across workers (each worker only needs its shard). Bandwidth used is the same, but memory drops linearly with world size.**

```python
def ddp_comm_per_step(param_count, dtype_bytes=2, world_size=8):
    activation_size = param_count * dtype_bytes
    ring_factor = 2 * (world_size - 1) / world_size
    return activation_size * ring_factor
```

For 70B params at fp16, world_size=64: `2 · 63/64 · 140GB ≈ 275 GB per rank per step`. Prohibitive without a fast interconnect.

## What to say aloud

> "Data parallel replicates the model across workers, splits the batch, and all-reduces gradients before the optimizer step. Every worker ends up with identical params. Communication is gradient-sized per step, once at the end of backward. Simple, but doesn't reduce memory — every worker still holds the full model. That's why FSDP/ZeRO-3 exists: it reduce-scatters gradients so each worker only stores its shard, then all-gathers params before each forward. Same total bandwidth as DDP but memory scales inversely with world size."

## Full version

Full multiprocessing simulation with actual concurrent processes lives in `movements/data_parallel_simulated/` (post-interview target).
