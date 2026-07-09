# Tensor Parallel — column-then-row pair (small)

The core TP pattern in Megatron-LM: pair a column-parallel linear with a row-parallel linear. Only one all-reduce per pair. This is the muscle-memory version — simulated in numpy, no multiprocessing.

Total: ~45 min.

## The pattern

For a 2-linear MLP `y = x @ W1 @ W2` (bias-free for simplicity), split across 2 workers:

**Column-parallel W1** (`d_model → d_ff`):
- Split W1 by *columns*: `W1_local = W1[:, worker_id * d_ff/2 : (worker_id+1) * d_ff/2]`
- Each worker computes: `h_local = x @ W1_local`, shape `(B, T, d_ff/2)`
- **No comm needed** — output is naturally split across workers.

**Row-parallel W2** (`d_ff → d_model`):
- Split W2 by *rows*: `W2_local = W2[worker_id * d_ff/2 : (worker_id+1) * d_ff/2, :]`
- Each worker computes: `y_partial = h_local @ W2_local`, shape `(B, T, d_model)`
- **All-reduce** `y_partial` across workers (sum). Result matches the full matmul.

Total communication per pair: **one all-reduce** of the activations, size `B · T · d_model`. Not per-linear.

## Exercise 1 — Implement the pair (25 min)

```python
def column_parallel(x, W_local):
    """
    x: (B, T, d_model)
    W_local: (d_model, d_local)   where d_local = d_ff / world_size
    Returns: (B, T, d_local)
    """
    return x @ W_local

def row_parallel(h_local, W_local):
    """
    h_local: (B, T, d_local)
    W_local: (d_local, d_model)
    Returns: (B, T, d_model) — the PARTIAL sum. Caller does all-reduce.
    """
    return h_local @ W_local

def simulate_tp_mlp(x, W1, W2, world_size):
    """
    Simulate world_size workers each computing their local slice of a
    column-parallel W1 + row-parallel W2 pair.
    
    W1: (d_model, d_ff)
    W2: (d_ff, d_model)
    
    Returns: (B, T, d_model) — the sum of all workers' y_partial.
    """
    d_ff = W1.shape[1]
    assert d_ff % world_size == 0
    d_local = d_ff // world_size
    
    y_partial_sum = np.zeros_like(x @ W1 @ W2)  # will accumulate
    for rank in range(world_size):
        W1_local = W1[:, rank * d_local : (rank + 1) * d_local]
        W2_local = W2[rank * d_local : (rank + 1) * d_local, :]
        h_local = column_parallel(x, W1_local)
        y_partial = row_parallel(h_local, W2_local)
        y_partial_sum += y_partial   # simulates the all-reduce
    return y_partial_sum
```

## Test

```python
np.random.seed(0)
B, T, d_model, d_ff = 2, 4, 16, 32
x = np.random.randn(B, T, d_model)
W1 = np.random.randn(d_model, d_ff)
W2 = np.random.randn(d_ff, d_model)

y_single = x @ W1 @ W2
y_tp = simulate_tp_mlp(x, W1, W2, world_size=4)

assert np.allclose(y_single, y_tp, atol=1e-6)
print("✓ TP matches single-worker computation")
```

## Exercise 2 — Communication analysis (10 min)

Compute bytes per all-reduce for one forward pass:

```python
def tp_comm_per_layer(B, T, d_model, world_size, dtype_bytes=2):
    """
    One all-reduce of the activations, size (B, T, d_model), 2 bytes fp16.
    Ring all-reduce: total traffic per rank ≈ 2 · (world_size - 1) / world_size · size.
    Returns bytes traversed per rank per all-reduce.
    """
    activation_bytes = B * T * d_model * dtype_bytes
    ring_factor = 2 * (world_size - 1) / world_size
    return activation_bytes * ring_factor
```

For a Llama-70B block, B=1, T=4096, d_model=8192, TP=8, fp16:
```
per-rank all-reduce ≈ 2 · 7/8 · (1 · 4096 · 8192 · 2) ≈ 117 MB
```

That's per all-reduce, per block. 80 blocks → 9.4 GB per forward. **Per-token comm.**

## What to say aloud (~90 sec)

> "Tensor parallel splits a linear layer across GPUs. Column-parallel splits W1 by columns — each GPU computes a slice of the output. Row-parallel splits W2 by rows — each GPU computes a partial output that needs to be all-reduced. The Megatron pattern pairs them: column-parallel W1 feeds directly into row-parallel W2 with no comm in between, and only one all-reduce at the end. This gives one all-reduce per Attention/FFN block. For Llama-70B at TP=8, that's ~117 MB per all-reduce per block per forward step. TP requires high-bandwidth intra-node interconnect (NVLink) because all-reduces are frequent and latency-sensitive; that's why TP within a node + PP across nodes is the standard pattern."

## Full version

Full multi-process simulation with actual concurrency lives in `movements/tensor_parallel_simulated/` (post-interview target).
