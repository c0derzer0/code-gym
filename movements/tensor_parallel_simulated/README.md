# Tensor Parallel simulated (full)

Multi-process Megatron-LM-style TP. Column-parallel + row-parallel linear pair. All-reduce activations. Verify: trained model matches single-worker training on the same data.

Total: **2 hr.** Prerequisite: `tp_column_row_pair` (numpy sim) done.

## What you'll build

An MLP `y = ReLU(x @ W1) @ W2` distributed across N workers with real multiprocessing.

```python
def train_tp(world_size, X, y, epochs, lr):
    """
    Spawn world_size workers. Each holds:
      - Column-parallel W1_local: (d_model, d_ff / world_size)
      - Row-parallel W2_local: (d_ff / world_size, d_model)
    
    Same batch on every worker. Forward computes partial y, all-reduce
    to get final y. Backward computes local grads, all-reduce for W1_local
    (column-parallel needs it), no all-reduce for W2_local (row-parallel
    accumulates naturally).
    """
    ...
```

## Exercises

### Exercise 1 — Forward + backward derivation (30 min)

Write it out:

**Forward:**
```
x     : (B, D)                     replicated across workers
h_i   = x @ W1_local[i]            (B, d_ff/N) per worker
h_act = ReLU(h_i)                  local
y_i   = h_act @ W2_local[i]        (B, D) partial — needs all-reduce
y     = all_reduce(y_i)            (B, D) full
```

**Backward:**
```
grad_y  : (B, D)                   replicated (all workers see full loss grad)
grad_y_i = grad_y                  each worker takes the full grad
grad_W2_local[i] = h_act[i].T @ grad_y_i
grad_h_act[i]    = grad_y_i @ W2_local[i].T   (B, d_ff/N)
grad_h[i]        = grad_h_act[i] * (h_i > 0)   local ReLU backward
grad_W1_local[i] = x.T @ grad_h[i]
grad_x_partial[i]= grad_h[i] @ W1_local[i].T   (B, D) partial — needs all-reduce
grad_x           = all_reduce(grad_x_partial)  (B, D) — used to propagate to prev layer
```

**Comm per forward+backward**: 2 all-reduces of size `(B, D)`.

### Exercise 2 — Multiprocessing implementation (60 min)

```python
def worker(rank, world_size, X, y, epochs, lr, model_params_channel, all_reduce_channels):
    """
    Worker process. Holds its shard of W1, W2.
    Communicates via multiprocessing pipes for all-reduce.
    """
    # Initialize local W1_local, W2_local (each rank uses a deterministic seed)
    torch.manual_seed(42)
    W1_full = torch.randn(D, d_ff) * 0.1
    W2_full = torch.randn(d_ff, D) * 0.1
    d_local = d_ff // world_size
    W1_local = W1_full[:, rank * d_local : (rank + 1) * d_local]
    W2_local = W2_full[rank * d_local : (rank + 1) * d_local, :]
    
    for epoch in range(epochs):
        # Forward
        h_i = X @ W1_local
        h_act = h_i.clip(min=0)  # ReLU
        y_i = h_act @ W2_local
        y_partial = my_all_reduce(y_i, rank, world_size, all_reduce_channels)  # sum
        
        # Loss (only rank 0 computes it for logging)
        if rank == 0:
            loss = ((y_partial - y_true) ** 2).mean()
            print(f"epoch {epoch}: loss = {loss:.4f}")
        
        # Backward
        grad_y = 2 * (y_partial - y_true) / len(X)  # every worker has the full grad
        grad_W2_local = h_act.T @ grad_y
        grad_h_act = grad_y @ W2_local.T
        grad_h_i = grad_h_act * (h_i > 0)
        grad_W1_local = X.T @ grad_h_i
        # grad_x doesn't matter here (no more layers)
        
        # Step (local)
        W1_local -= lr * grad_W1_local
        W2_local -= lr * grad_W2_local
    
    return W1_local, W2_local
```

### Exercise 3 — Verify against single-worker (30 min)

```python
# Train single-worker
np.random.seed(0)
X = torch.randn(200, 8)
y_true = torch.randn(200, 8)
W1_full = torch.randn(8, 32) * 0.1  # same init
W2_full = torch.randn(32, 8) * 0.1
for epoch in range(100):
    h = (X @ W1_full).clip(min=0)
    pred = h @ W2_full
    loss = ((pred - y_true) ** 2).mean()
    grad_pred = 2 * (pred - y_true) / len(X)
    grad_W2 = h.T @ grad_pred
    grad_h = (grad_pred @ W2_full.T) * (h > 0)  # ReLU backward on positives
    grad_W1 = X.T @ grad_h
    W1_full -= 0.01 * grad_W1
    W2_full -= 0.01 * grad_W2

# Train TP with world_size=4
W1_locals, W2_locals = train_tp(world_size=4, X=X, y=y_true, epochs=100, lr=0.01)

# Reassemble the TP params
W1_reassembled = torch.cat(W1_locals, dim=1)
W2_reassembled = torch.cat(W2_locals, dim=0)

assert torch.allclose(W1_full, W1_reassembled, atol=1e-4)
assert torch.allclose(W2_full, W2_reassembled, atol=1e-4)
print("✓ TP matches single-worker training")
```

## Watch out for

1. **Deterministic init across workers**: every worker must init the same `W1_full`, then slice. Otherwise you can't verify equivalence.
2. **All-reduce implementation**: implement your own via pipes. Basic ring all-reduce or naive gather-broadcast. Even simple works for correctness, real speed needs NCCL.
3. **Same batch on every worker**: TP replicates the batch. Don't split it (that's DP).
4. **Backward for `x`**: even if you don't use `grad_x`, it needs the all-reduce for row-parallel + column-parallel pattern. In a multi-layer network, it matters.
5. **Cross-worker gradient consistency**: since every worker sees the same batch and the same `grad_y`, their `grad_W1_local, grad_W2_local` are already the correct shard — no gradient all-reduce needed for W.

## What to say aloud (~2 min)

> "Tensor parallelism splits the model's weight matrices across workers. Megatron-LM's column-then-row pattern is the canonical approach: column-parallel splits W1 by output columns, so each worker computes a slice of the intermediate activation with no comm. Row-parallel splits W2 by input rows, so each worker computes a partial output that needs to be all-reduced. Pairing them means only one all-reduce per Attention/FFN block instead of two.
>
> Every worker sees the full batch. Communication is activation-sized per all-reduce, per block, per forward — for Llama-70B with TP=8, that's ~117 MB per all-reduce per block. High-frequency, latency-sensitive, so TP needs NVLink or NVSwitch — high-bandwidth intra-node interconnect. That's why the standard is TP within a node + pipeline parallel across nodes.
>
> Backward pass also does one all-reduce per block for the input gradient. Total comm per forward+backward per block: 2 all-reduces of activation-sized tensors. About 2× the memory bandwidth pressure of the compute itself."

## References

- Shoeybi et al. 2019, "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
- Real implementation: `megatron/mpu/` in NVIDIA/Megatron-LM repo
