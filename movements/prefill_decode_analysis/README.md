# Prefill vs decode — compute analysis

The two fundamentally different inference regimes. Prefill is compute-bound; decode is memory-bound. Understanding this asymmetry drives every serving optimization.

Total: ~30 min. Uses your existing KV cache.

## The two regimes

**Prefill** — process the entire input prompt at once. All T tokens go through the model in one forward pass.
- Compute: `O(T · P)` (P = param count)
- Attention: `O(T²)` per layer
- **High arithmetic intensity — compute-bound**

**Decode** — after prefill, generate one token at a time, using KV cache.
- Each step processes 1 token
- Compute: `O(1 · P)` = `O(P)` per step
- Attention: `O(T · d)` per step (attend one query over cached T keys)
- **Low arithmetic intensity — memory-bound**

**The critical insight**: Decode reads all P parameters from HBM per token. If P = 70B fp16 = 140 GB, and HBM = 2 TB/s, minimum per-token latency is 70 ms. **You can't beat this without batching, quantization, or speculative decoding.**

## Exercise 1 — Measure arithmetic intensity (15 min)

```python
def prefill_ai(P: int, T: int, dtype_bytes: int = 2) -> float:
    """
    Approximate arithmetic intensity for prefill of a T-token prompt.
    
    FLOPs: ~2 · T · P (matmul dominates: T output positions, each needs 2 FLOPs per param)
    HBM reads: params (P · dtype_bytes) + activations (T · d · L · dtype_bytes)
    HBM writes: activations
    
    Simplification: assume attention/normalization negligible vs matmuls.
    Reads dominated by param load: P · dtype_bytes (one full pass through weights).
    
    Returns FLOP / byte.
    """
    flops = 2 * T * P
    bytes_hbm = P * dtype_bytes   # dominant: read all weights once
    return flops / bytes_hbm

def decode_ai(P: int, dtype_bytes: int = 2) -> float:
    """
    Same math but T=1: process only one token.
    """
    return prefill_ai(P, T=1, dtype_bytes=dtype_bytes)

def kv_cache_ai(T: int, d_model: int, n_layers: int, dtype_bytes: int = 2) -> float:
    """
    Per-decode-step KV cache read:
    - Read all L layers' KV cache: 2 · L · T · d_model · dtype_bytes bytes
    - Attention FLOPs: 2 · T · d_model per layer, sum L layers = 2 · L · T · d_model
    - So the KV cache alone has AI = 1/dtype_bytes  ← extremely low
    """
    ...
```

Run for Llama-70B (P=70e9, d_model=8192, L=80):

```python
# Prefill on a 4096-token prompt
ai_prefill = prefill_ai(70e9, T=4096)
print(f"prefill T=4096: AI = {ai_prefill:.0f} FLOP/byte")
# expected: ~4000, compute-bound (well above A100's 156 ridge)

# Decode one token (with prefill already done)
ai_decode = decode_ai(70e9)
print(f"decode T=1: AI = {ai_decode:.0f} FLOP/byte")
# expected: ~1, deeply memory-bound

# KV cache during decode
ai_kv = kv_cache_ai(T=4096, d_model=8192, n_layers=80)
print(f"kv cache: AI = {ai_kv:.2f} FLOP/byte")
# expected: 0.5, worst possible
```

## Exercise 2 — Throughput implications (15 min)

```python
def prefill_time(P: int, T: int, peak_tflops: float = 312) -> float:
    """
    Compute-bound: time = FLOPs / peak_tflops
    Returns seconds.
    """
    flops = 2 * T * P
    return flops / (peak_tflops * 1e12)

def decode_time_per_token(P: int, peak_hbm_gb_s: float = 2000) -> float:
    """
    Memory-bound: time = param_bytes / peak_bw
    Returns seconds per token.
    """
    param_bytes = P * 2  # fp16
    return param_bytes / (peak_hbm_gb_s * 1e9)
```

For Llama-70B on A100 (312 TFLOP/s fp16, 2 TB/s HBM):

```python
prefill_s = prefill_time(70e9, T=4096)
decode_ms = decode_time_per_token(70e9) * 1000

print(f"prefill 4K tokens: {prefill_s:.2f} s")     # ~1.8 s
print(f"decode per token:  {decode_ms:.1f} ms")    # ~70 ms
print(f"decode throughput: {1/(decode_ms/1000):.1f} tok/s")  # ~14 tok/s
```

**The 70 ms/token bound is why batching decode is so critical**: batching amortizes the param load across N sequences, giving N× throughput at the same per-token latency (until memory becomes the bottleneck).

## What to say aloud (~90 sec)

> "Prefill and decode have opposite bottlenecks. Prefill processes a full prompt in one forward — high arithmetic intensity, compute-bound. Decode processes one token per step reading the entire model from HBM — extremely low arithmetic intensity, memory-bound. For a 70B model on A100, a single decode step reads 140 GB of weights and runs 140 GFLOPs of compute. Time is memory-limited: 140GB / 2TB/s = 70 ms per token, giving ~14 tok/s for a single sequence.
>
> This asymmetry drives serving design. To hit real throughput you must batch decode — the memory bandwidth cost of loading weights is paid once and amortized across N sequences in the batch. Alternative wins: quantization (halves memory bandwidth), speculative decoding (verifies multiple tokens per forward), and continuous batching (keep the batch full as sequences finish)."

## References

- vLLM paper — PagedAttention + continuous batching
- SGLang / LMDeploy — modern serving frameworks addressing this asymmetry
