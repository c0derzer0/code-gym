# GPU compute model — SMs, blocks, kernels, roofline

Not a writeup — five concrete exercises. Each forces you to see a number change based on a decision you make (tile size, fusion vs no fusion, block size, etc.). Doing beats reading for internalizing this material.

Total: ~75 min. Each exercise is 10-15 min.

## Setup

Every exercise lives in `attempts/attempt_1.py` as a separate function. At the end of the file, `main()` runs all five and prints results.

Common hardware spec (A100):
```python
A100 = {
    "peak_flops_fp16": 312e12,   # 312 TFLOP/s
    "peak_hbm_bw": 2e12,          # 2 TB/s
    "n_sms": 108,
    "sram_per_sm_bytes": 192 * 1024,
    "max_warps_per_sm": 64,
    "max_regs_per_sm": 65536,
    "max_shared_per_block_bytes": 48 * 1024,  # typical
}
```

## Exercise 1 — HBM traffic calculator (10 min)

Implement a function that returns bytes read, bytes written, and FLOPs for common ops. This is the counting muscle every roofline calculation depends on.

```python
def op_cost(op: str, shapes: dict, dtype_bytes: int = 2) -> dict:
    """
    Return {'reads': int, 'writes': int, 'flops': int, 'ai': float}
    for the named op. 'ai' = flops / (reads + writes).
    
    Supported ops:
      - 'matmul':      shapes = {'M', 'K', 'N'}    # A (M,K) @ B (K,N) → (M,N)
      - 'layernorm':   shapes = {'B', 'T', 'D'}    # normalize last dim
      - 'elementwise': shapes = {'N'}              # e.g. relu, N elements in/out
      - 'attention':   shapes = {'B', 'H', 'T', 'd_head'}  # QK^T/√d, softmax, @V
    """
    ...
```

**Concrete numbers you should be able to derive by hand for a matmul (M=K=N=1024, fp16):**
- reads = `2 · 1024 · 1024 · 2 = 4 MB` (A and B)
- writes = `1024 · 1024 · 2 = 2 MB` (C)
- FLOPs = `2 · M · N · K = 2 · 1024³ ≈ 2.1 G`
- AI = `2.1G / 6M ≈ 350 FLOP/byte`

Verify your function matches these hand calculations before moving on.

## Exercise 2 — Roofline classifier (15 min)

Given hardware peak_flops and peak_bw, classify each op as memory-bound or compute-bound. Print achievable throughput assuming perfectly optimal implementation.

```python
def roofline(op_cost: dict, hw: dict) -> dict:
    """
    Return {'bound': 'memory' | 'compute', 'peak_gflops': float, 'is_hbm_bound': bool}
    
    ridge_point = hw['peak_flops_fp16'] / hw['peak_hbm_bw']
    if op_cost['ai'] < ridge_point → memory-bound (bandwidth is bottleneck)
    if op_cost['ai'] >= ridge_point → compute-bound (FLOPs is bottleneck)
    
    Achievable throughput:
      if memory-bound: peak_bw * ai
      if compute-bound: peak_flops
    """
    ...
```

Run on these 6 ops with A100 numbers, print a table:

| Op | shape | AI (FLOP/byte) | bound | achievable TFLOP/s |
|---|---|---|---|---|
| matmul (fp16) | M=K=N=1024 | ~350 | compute | 312 |
| matmul (fp16) | M=1, K=N=8192 | ~2 | memory | 4 |
| layernorm | B=8, T=2048, D=4096 | ~0.5 | memory | 1 |
| attention (fp16) | B=8, H=32, T=2048, d_head=128 | ~10 | memory | 20 |
| elementwise (relu) | N=100M | 0.25 | memory | 0.5 |
| kv_cache_read | B=8, T=4096, d=8192 | 0 | memory | 0 |

**The lesson:** A100's ridge point is `312e12 / 2e12 = 156 FLOP/byte`. Anything below that is memory-bound. **Most LLM ops sit far below the ridge.** That's why kernel optimization = minimize HBM traffic, not add FLOPs.

Also notice: `M=1` matmul (single query decode step) is 100× less efficient than a big matmul. This is why batching matters for inference.

## Exercise 3 — Tiled matmul with memory counting (15 min)

Implement matmul as tiled, count HBM reads. Sweep tile sizes and see AI change.

```python
def tiled_matmul(A, B, block_M, block_N, block_K):
    """
    Compute A @ B via tiling.
    Track hbm_bytes = total bytes 'loaded from HBM' (each tile load counts).
    
    In a real GPU kernel:
      - Output tile is (block_M, block_N) — stored in registers/accumulator
      - Loop over K tiles: load A_tile (block_M, block_K), B_tile (block_K, block_N)
      - Accumulate partial matmul into output
    
    Returns (C, hbm_bytes).
    """
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))
    hbm_bytes = 0
    
    # ... your tiling logic ...
    
    return C, hbm_bytes
```

**Verify** first: `np.allclose(tiled_matmul(A, B, 32, 32, 16)[0], A @ B, atol=1e-5)`.

Then **sweep** tile sizes:

```python
for bs in [8, 16, 32, 64, 128]:
    C, bytes_loaded = tiled_matmul(A, B, bs, bs, bs)
    flops = 2 * M * N * K
    ai = flops / bytes_loaded
    print(f"tile={bs}: AI = {ai:.1f} FLOP/byte, HBM = {bytes_loaded / 1e6:.2f} MB")
```

**Expected output**: AI grows as tile size grows. Larger tiles reuse loaded data more times → higher AI.

**But** tile size is bounded by shared memory (192 KB per SM on A100). At `bs = 128, fp16`, one tile is `128 * 128 * 2 = 32 KB`. You have A_tile + B_tile + C accumulator in SRAM — 96 KB. Fits. At `bs = 256`, one tile is 128 KB. Two tiles = 256 KB. Doesn't fit. **This is why block size is limited by shared memory in real kernels.**

## Exercise 4 — Fusion demonstration (10 min)

Compare unfused vs fused `y = relu(x + b)`.

```python
def relu_add_unfused(x, b):
    """
    Two 'kernels' in sequence:
      k1:  tmp = x + b        (read x, read b, write tmp)
      k2:  y = relu(tmp)      (read tmp, write y)
    
    Return (y, hbm_bytes).
    """
    ...

def relu_add_fused(x, b):
    """
    One kernel:  y = relu(x + b)  (read x, read b, write y — intermediate stays in registers)
    Return (y, hbm_bytes).
    """
    ...

# Compare:
x = np.random.randn(1_000_000).astype(np.float16)
b = np.random.randn(1_000_000).astype(np.float16)

y1, bytes_unfused = relu_add_unfused(x, b)
y2, bytes_fused = relu_add_fused(x, b)

assert np.allclose(y1, y2)
print(f"unfused HBM: {bytes_unfused / 1e6:.1f} MB")
print(f"fused HBM:   {bytes_fused / 1e6:.1f} MB")
print(f"speedup (bw-bound): {bytes_unfused / bytes_fused:.2f}×")
```

**Expected**: unfused is ~1.67× more HBM traffic (3 reads + 1 write vs 2 reads + 1 write). Since this op is bandwidth-bound, that's the actual speedup you'd see.

**The lesson**: every "extra kernel" is an HBM round-trip. This is why PyTorch's `torch.compile`, torch2 fx compilers, Triton fusion, etc. all exist — to eliminate these round-trips.

## Exercise 5 — Occupancy calculator (10 min)

Given per-block resource usage, calculate how many blocks can run concurrently on one SM.

```python
def blocks_per_sm(regs_per_thread: int,
                  shared_per_block_bytes: int,
                  threads_per_block: int,
                  hw: dict) -> dict:
    """
    Compute how many blocks can co-reside on one SM, limited by three constraints:
      - warps_limit    = max_warps_per_sm // (threads_per_block // 32)
      - registers_limit = max_regs_per_sm // (regs_per_thread * threads_per_block)
      - shared_limit   = sram_per_sm_bytes // shared_per_block_bytes  (if > 0)
    
    Return {'blocks_per_sm': int, 'bottleneck': str, 'occupancy_frac': float}
    where bottleneck is which constraint bound the answer.
    """
    ...
```

Run these three scenarios:

```python
# Scenario A: modest kernel
print(blocks_per_sm(regs_per_thread=32, shared_per_block_bytes=8*1024,
                    threads_per_block=256, hw=A100))

# Scenario B: register-heavy kernel (e.g. big block-level accumulator)
print(blocks_per_sm(regs_per_thread=128, shared_per_block_bytes=8*1024,
                    threads_per_block=256, hw=A100))

# Scenario C: shared-memory-heavy kernel (e.g. big tile in shared memory)
print(blocks_per_sm(regs_per_thread=32, shared_per_block_bytes=40*1024,
                    threads_per_block=256, hw=A100))
```

**Expected**:
- A: ~4-8 blocks/SM. Well occupied.
- B: 2 blocks/SM (registers become the bottleneck).
- C: 4 blocks/SM (shared memory becomes the bottleneck).

**The lesson**: increasing tile size (more shared memory per block) or complexity (more registers per thread) can **reduce** the number of blocks that co-reside on an SM. Fewer blocks = fewer warps in flight = less latency hiding = potentially worse throughput. This is why block-size tuning is a real thing in kernel writing.

## The 5 lessons, memorized after this movement

1. **HBM traffic is countable.** You now have a mechanical function for it, not vibes.
2. **A100's ridge point is 156 FLOP/byte.** Almost every LLM op is below this. Memory-bound is the default.
3. **Tiling raises arithmetic intensity by reusing loaded data.** Bigger tiles = higher AI, up to the point shared memory can't hold them.
4. **Kernel fusion saves an HBM round-trip per fused op.** Every extra kernel launched is bandwidth wasted.
5. **Occupancy is capped by three constraints** — warps, registers, shared memory. Kernel tuning navigates all three.

## Test (all should pass)

```python
if __name__ == "__main__":
    # Exercise 1
    r = op_cost('matmul', {'M': 1024, 'K': 1024, 'N': 1024}, dtype_bytes=2)
    assert r['reads'] + r['writes'] == (2 + 1) * 1024 * 1024 * 2
    assert r['flops'] == 2 * 1024 ** 3

    # Exercise 3
    A = np.random.randn(64, 96).astype(np.float32)
    B = np.random.randn(96, 128).astype(np.float32)
    C_tiled, hbm = tiled_matmul(A, B, 32, 32, 16)
    assert np.allclose(C_tiled, A @ B, atol=1e-4)
    assert hbm > 0

    # Exercise 4
    x = np.random.randn(1000).astype(np.float16)
    b = np.random.randn(1000).astype(np.float16)
    y_u, hbm_u = relu_add_unfused(x, b)
    y_f, hbm_f = relu_add_fused(x, b)
    assert np.allclose(y_u, y_f)
    assert hbm_u > hbm_f

    # Exercise 5
    r = blocks_per_sm(32, 8*1024, 256, A100)
    assert 1 <= r['blocks_per_sm'] <= 16

    print("all exercises pass")
```

## What to say aloud (~90 seconds, unified answer to any GPU-kernel question)

> "GPU kernels are dispatched as a grid of blocks; each block runs on one SM (a Streaming Multiprocessor). Within a block, threads are grouped into 32-thread warps that execute in lockstep. Each SM has ~192 KB of on-chip SRAM shared across its blocks, and thousands of registers per thread.
>
> Memory hierarchy from fast to slow: registers → shared memory (SRAM) → L2 cache → HBM. HBM bandwidth is the bottleneck for most ML ops. A100 has 2 TB/s HBM and 312 TFLOP/s compute, giving a ridge point of 156 FLOP/byte. Anything below is memory-bound.
>
> Optimization is fundamentally about reducing HBM traffic: tile data to reuse it in SRAM, fuse elementwise ops to avoid round-trips, batch small ops to amortize fixed costs. Roofline analysis quantifies where you sit: an op's arithmetic intensity determines whether adding compute helps (compute-bound) or is wasted (memory-bound). This is why Flash Attention, kernel fusion, and quantization all matter — they all move ops toward higher AI or lower memory usage per FLOP."

## References (after timer)

- NVIDIA A100 whitepaper — the concrete numbers
- Simon Boehm's blog: "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance" — hands-on kernel walkthrough
- Roofline paper (Williams, Waterman, Patterson, 2009) — the original methodology
- Triton tutorial 03 (fused softmax) — bridges concept to real Triton code
- PMPP (Programming Massively Parallel Processors), Hwu / Kirk, ch. 3-5

## Attempt header template

```python
# Attempt N — YYYY-MM-DD
# Started:
# Exercises passing: 1/5, 2/5, ...
# Notes:
```
