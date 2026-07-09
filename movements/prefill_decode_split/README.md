# Prefill / decode split (full)

Full separation of prefill and decode paths — plus chunked prefill (the modern serving optimization). Builds on `kv_cache` and `prefill_decode_analysis`.

Total: **90 min.** Prerequisite: `kv_cache` done, `prefill_decode_analysis` done.

## What you'll build

A minimal inference loop with:
- Separate `prefill()` and `decode()` methods on your MHA-with-KV-cache
- Chunked prefill: split a long prompt into chunks to fit within compute budget
- A "batch scheduler" that mixes prefill and decode requests in a single forward pass (the vLLM continuous batching pattern)

```python
class InferenceEngine:
    def __init__(self, model, max_seq_len, max_batch_size):
        self.model = model  # your transformer
        self.caches = {}    # seq_id → KV cache dict
    
    def prefill(self, seq_id: int, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Process the entire prompt in one forward pass.
        Populate KV cache. Return output logits for the last position.
        """
        ...
    
    def prefill_chunked(self, seq_id: int, prompt_tokens: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """
        Process prompt in chunks of chunk_size tokens, extending the cache incrementally.
        Same final result as prefill(), but caps peak compute.
        """
        ...
    
    def decode(self, seq_id: int, new_token: int) -> torch.Tensor:
        """
        Single-token forward with KV cache. Return logits.
        """
        ...
    
    def mixed_batch(self, prefill_reqs: list, decode_reqs: list) -> dict:
        """
        Continuous batching: run all decode requests + as many prefill chunks
        as fit in the compute budget in a single forward pass.
        Return {seq_id: logits} for each request served.
        """
        ...
```

## Exercises

### Exercise 1 — Basic prefill vs decode (30 min)

Verify: `prefill(prompt) followed by T decode()s` gives same output as `prefill(prompt + T generated tokens)`. Correctness proof.

Also measure throughput:

```python
model = build_toy_transformer(n_layers=4, d_model=64, n_heads=8, vocab=1000)
engine = InferenceEngine(model, max_seq_len=1024, max_batch_size=1)

# Prefill 512 tokens
tokens = torch.randint(0, 1000, (512,))
t0 = time.time()
out_prefill = engine.prefill(seq_id=0, prompt_tokens=tokens)
prefill_time = time.time() - t0

# Decode 64 tokens
t0 = time.time()
for _ in range(64):
    logits = engine.decode(seq_id=0, new_token=torch.randint(0, 1000, (1,)).item())
decode_time = time.time() - t0

print(f"prefill (512 tok): {prefill_time*1000:.1f} ms → {512/prefill_time:.0f} tok/s")
print(f"decode (64 tok): {decode_time*1000:.1f} ms → {64/decode_time:.0f} tok/s")
# Expected: decode tok/s much lower than prefill tok/s — memory-bound
```

### Exercise 2 — Chunked prefill (30 min)

Split a long prompt (e.g., 2048 tokens) into chunks of 256. Process each chunk as a mini-prefill that extends the KV cache.

Why: real serving systems can't allocate compute for a 32k-token prompt in one shot without stalling other requests. Chunked prefill lets them interleave.

Test:
```python
tokens = torch.randint(0, 1000, (2048,))

# Full prefill
out_full = engine.prefill(seq_id=0, prompt_tokens=tokens)
engine.caches.clear()

# Chunked prefill (chunks of 256)
out_chunked = engine.prefill_chunked(seq_id=1, prompt_tokens=tokens, chunk_size=256)

assert torch.allclose(out_full, out_chunked, atol=1e-5)
```

### Exercise 3 — Mixed batch (continuous batching) (30 min)

Run a single forward that processes:
- 3 decode requests (each T_new=1)
- 1 prefill chunk (256 new tokens)

Total sequence input: `3 + 256 = 259` tokens. But different lengths of past context.

Requires: variable-length attention (from `paged_attention_full`), or padding + masking (simpler for this exercise).

The point: instead of alternating prefill batch → decode batch → prefill batch, mix them. Keeps the GPU busy on both compute-bound and memory-bound work.

## Watch out for

1. **KV cache is per-sequence**: multi-seq batches need per-seq caches OR paged attention.
2. **Chunked prefill boundary**: the last token of chunk N and first of chunk N+1 must be able to attend to each other. Get the position IDs right.
3. **Loss of causal mask across chunks**: sliced correctly, causality holds. Draw it out for chunk_size=4, prompt_length=8.
4. **Continuous batching order matters**: latency-sensitive decodes should preempt long prefills. Real schedulers have priority queues.
5. **Chunked prefill isn't free**: extra overhead per chunk (kernel launch, cache setup). At small chunk sizes, throughput drops. Typical range: chunk_size = 512-2048.

## What to say aloud (~2 min)

> "Prefill and decode have opposite compute characteristics. Prefill processes a whole prompt in one forward pass — high arithmetic intensity, compute-bound. Decode processes one token per step reading the entire model from HBM per step — memory-bound. On A100, decode is limited to ~14 tokens/second per sequence for a 70B model, because you're reading 140 GB of weights per token at 2 TB/s.
>
> Chunked prefill splits a long prompt into fixed-size chunks (say 512 tokens each), processed sequentially. Each chunk extends the KV cache. Same final result as one big prefill, but caps peak GPU compute — letting the scheduler interleave decode requests without long-tail latency spikes.
>
> Continuous batching (vLLM's key innovation) mixes prefill and decode in the same forward pass. New sequences start prefill while running sequences decode, all in one batch. Combined with paged attention (variable-length KV), this pattern hits near-optimal GPU utilization: compute-bound work from prefill amortizes with memory-bound decode over the same kernel launch."

## References

- vLLM paper (Kwon et al. 2023) — continuous batching + PagedAttention
- SGLang paper — RadixAttention for prefix caching
- Chunked prefill: originally proposed in DeepSpeed-Inference / adopted by vLLM
