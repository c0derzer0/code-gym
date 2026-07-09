# Paged Attention (full)

Complete paged KV cache + attention forward pass. Builds on `paged_kv_cache` (block allocator + block table). Adds: attention that reads through the block table, batched processing of variable-length sequences, memory accounting.

Total: **90 min - 2 hr.** Prerequisite: `paged_kv_cache` exercises 1-2 done.

## What you'll build

Full paged attention for a **batch of sequences with different lengths**, sharing a single physical KV block pool.

```python
class PagedAttention:
    def __init__(self, num_blocks, block_size, n_heads, d_head, d_model):
        self.allocator = BlockAllocator(num_blocks, block_size, n_heads, d_head)
        # Q, K, V, out projections (nn.Linear)
        ...
    
    def forward_prefill(self, tokens: list[torch.Tensor], sequences: list[Sequence]) -> torch.Tensor:
        """
        For each sequence:
          - Compute Q, K, V for the prompt tokens
          - Store K, V into paged blocks
          - Compute attention (Q attends over own cache only)
        Returns: list of output tensors, one per sequence (variable T)
        """
        ...
    
    def forward_decode(self, next_tokens: torch.Tensor, sequences: list[Sequence]) -> torch.Tensor:
        """
        Single-token decode step for a batch of sequences.
        Each sequence's Q attends over its own cache (fetched via block table).
        - Compute Q, K, V for the single new token per sequence
        - Append K, V to the block table
        - Compute attention using the extended cache
        Returns: (B, d_model) — one output per sequence
        """
        ...
```

## Exercises

### Exercise 1 — Gather K, V via block table (30 min)

```python
def gather_kv_from_blocks(sequence, allocator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct sequence's K, V by reading through its block table.
    Returns (K, V) of shape (length, n_heads, d_head).
    """
    ...

# Test:
# Populate a sequence with T=17 tokens, block_size=8 → 3 blocks (last one partial)
# gather_kv_from_blocks reconstructs the same tensors you appended
```

### Exercise 2 — Batched decode with variable-length attention (60 min)

For a batch of 4 sequences with lengths `[3, 7, 5, 10]`:
- Each computes Q for its 1 new token: `Q_i` shape `(1, n_heads, d_head)`
- Each gathers its own K/V from block table: `K_i, V_i` shape `(len_i, n_heads, d_head)`
- Attention: `softmax(Q_i @ K_i^T / √d) @ V_i` for each seq
- Return: `(4, d_model)` (one output per seq)

**The tricky bit**: variable-length attention. You can't naively pad — that's what you're avoiding. Two options:
- **Flat concat**: `Q = cat([Q_0, Q_1, Q_2, Q_3])`, run attention per-seq via indexing (loop).
- **Nested tensor** (PyTorch feature): supports variable seq length in a batch.

Do the loop version first — it's easier to verify correctness.

### Exercise 3 — Memory accounting under sequence churn (30 min)

Simulate a realistic serving load:

```python
# Start with an empty allocator
allocator = BlockAllocator(num_blocks=1000, block_size=16, ...)

# Emulate 100 requests, each with a random prompt length + generation length
for step in range(100):
    # (a) with some probability, add a new sequence with a fresh prompt
    # (b) tick each active sequence forward by one token
    # (c) with some probability, "finish" a sequence and free its blocks
    ...

# Track: max blocks in use, blocks currently in use, waste ratio
```

Compare against a contiguous cache that would need to be sized for the maximum-ever concurrent sequences at maximum length. **You'll see paged gets ~5× more concurrent sequences at the same block pool size.**

## Tests

```python
np.random.seed(0)
d_model, n_heads = 32, 4
allocator = BlockAllocator(num_blocks=50, block_size=8, n_heads=n_heads, d_head=8)

# Sequence 1: prefill 12 tokens
seq1 = Sequence(0, allocator)
tokens1 = np.random.randn(12, n_heads, 8)
for t in tokens1:
    seq1.append(t, t)  # k=v for simplicity

# Sequence 2: prefill 5 tokens
seq2 = Sequence(1, allocator)
tokens2 = np.random.randn(5, n_heads, 8)
for t in tokens2:
    seq2.append(t, t)

# Both sequences should have their own block tables
assert len(seq1.block_table) == 2  # ceil(12/8) = 2
assert len(seq2.block_table) == 1  # ceil(5/8) = 1

# gather returns correct data
K1, _ = gather_kv_from_blocks(seq1, allocator)
assert K1.shape == (12, n_heads, 8)
assert np.allclose(K1, tokens1)

# Free and re-use
seq1.release()
assert allocator.num_free() == 50 - 1  # only seq2's block still allocated
```

## Watch out for

1. **Block table lookup order**: block `k` in the table covers logical positions `k * block_size : (k+1) * block_size`. Off-by-one bugs are common.
2. **Partial last block**: last block usually only partially filled. `gather_kv_from_blocks` must slice to `seq.length`, not fetch the whole last block.
3. **Free doesn't zero**: freed blocks are reclaimable but old data lingers until overwritten. Not a correctness bug for the tests, but worth noting.
4. **Concurrent sequences share the pool**: two sequences can't own the same block. Assert in allocator.
5. **Attention indexing**: with variable-length sequences, the causal mask must be per-sequence; there's no shared `(T, T)` mask.

## What to say aloud (~2 min)

> "Paged attention allocates KV cache in fixed-size blocks (typically 16 tokens each) from a shared physical pool. Each sequence has a block table — a list of block indices — that maps logical positions to physical locations. Attention gathers K and V via block table lookups instead of a contiguous read.
>
> During decode, each sequence's Q attends only over its own cached K, V, fetched through its block table. This eliminates the padding waste of contiguous batched attention — a variable-length batch of 32 sequences might use 40% of allocated memory in contiguous, versus 95% in paged. That memory saving translates to serving 3-5× more concurrent sequences at the same GPU memory.
>
> The block-table indirection is a small overhead, but it enables continuous batching (new sequences claim blocks as running sequences finish), prefix caching (multiple sequences can share the same prompt blocks by pointing to them), and dynamic memory management overall. It's why vLLM is 10× faster than naive HF transformers batching."

## References

- Kwon et al. 2023, "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- vLLM source code: `vllm/attention/backends/paged_attn.py`
