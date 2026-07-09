# Paged KV cache (vLLM PagedAttention)

Not a writeup — three exercises. By the end you can explain PagedAttention mechanically: block table, physical/logical mapping, why padding waste disappears.

Total: ~45 min. Pure numpy. Prerequisite: `kv_cache` movement done.

## The problem PagedAttention solves

Contiguous KV cache: allocate `(B, T_max, ...)`. Every sequence gets `T_max` slots regardless of actual length.

Batch of sequences with lengths `[5, 8, 3, 6]`, `T_max = 8`:
- Total allocated: `4 * 8 = 32` slots
- Actually used: `5 + 8 + 3 + 6 = 22` slots
- **Wasted: 10 slots (31%).** Padding waste.

At production scale (variable-length prompts, dynamically-arriving requests), waste often hits 60-80%.

**PagedAttention fix:** allocate KV cache in fixed-size **blocks**. Each sequence owns a list of blocks (its "block table"). When it grows, it claims a new block. When it finishes, its blocks return to a free pool. No padding.

## Exercise 1 — Block allocator with free pool (15 min)

Implement a simple block allocator.

```python
class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int, d_head: int, n_heads: int):
        """
        Physical storage: (num_blocks, block_size, n_heads, d_head)
        Maintain a set of free block indices.
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.d_head = d_head
        self.n_heads = n_heads
        self.k_pool = np.zeros((num_blocks, block_size, n_heads, d_head), dtype=np.float32)
        self.v_pool = np.zeros((num_blocks, block_size, n_heads, d_head), dtype=np.float32)
        self.free_blocks = set(range(num_blocks))
    
    def allocate(self) -> int:
        """Return a free block index, mark it in-use."""
        ...
    
    def free(self, block_idx: int):
        """Return a block to the free pool."""
        ...
    
    def num_free(self) -> int:
        return len(self.free_blocks)
```

Test:
```python
allocator = BlockAllocator(num_blocks=100, block_size=16, d_head=64, n_heads=8)
assert allocator.num_free() == 100
b0 = allocator.allocate()
b1 = allocator.allocate()
assert allocator.num_free() == 98
allocator.free(b0)
assert allocator.num_free() == 99
```

## Exercise 2 — Sequence with block table (15 min)

Each sequence tracks which blocks it owns.

```python
class Sequence:
    def __init__(self, seq_id: int, allocator: BlockAllocator):
        self.seq_id = seq_id
        self.allocator = allocator
        self.block_table = []       # list of block indices
        self.length = 0             # current token count
    
    def append(self, k_new: np.ndarray, v_new: np.ndarray):
        """
        k_new, v_new: (n_heads, d_head) — a single new token's K, V.
        Write into the current block. If block is full, allocate a new one.
        """
        block_size = self.allocator.block_size
        # position within the last block:
        offset = self.length % block_size
        if offset == 0:
            # need a new block
            new_block = self.allocator.allocate()
            self.block_table.append(new_block)
        current_block = self.block_table[-1]
        self.allocator.k_pool[current_block, offset] = k_new
        self.allocator.v_pool[current_block, offset] = v_new
        self.length += 1
    
    def gather_kv(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the sequence's full K, V from its block table.
        Returns K, V of shape (length, n_heads, d_head).
        """
        ...
    
    def release(self):
        """Free all blocks back to the pool."""
        ...
```

Test:
```python
allocator = BlockAllocator(num_blocks=10, block_size=4, d_head=8, n_heads=2)
seq = Sequence(seq_id=0, allocator=allocator)

# Add 10 tokens (needs 3 blocks: 4 + 4 + 2)
tokens = [np.random.randn(2, 8).astype(np.float32) for _ in range(10)]
for k in tokens:
    seq.append(k, k)  # use same k for both k and v for simplicity

assert seq.length == 10
assert len(seq.block_table) == 3        # ceil(10/4) = 3 blocks
assert allocator.num_free() == 7        # 10 - 3

# Gather and verify order preserved
K, V = seq.gather_kv()
assert K.shape == (10, 2, 8)
for i, k in enumerate(tokens):
    assert np.allclose(K[i], k)

seq.release()
assert allocator.num_free() == 10
```

## Exercise 3 — Batch of variable-length sequences: measure waste (15 min)

Compare contiguous vs paged for a realistic batch.

```python
def contiguous_memory(seq_lengths: list[int], d_head: int, n_heads: int, dtype_bytes: int = 2) -> dict:
    """
    Contiguous KV cache allocation: (B, T_max, n_heads, d_head), 2× for K + V.
    Returns {'allocated_bytes', 'used_bytes', 'waste_frac'}.
    """
    B = len(seq_lengths)
    T_max = max(seq_lengths)
    allocated = 2 * B * T_max * n_heads * d_head * dtype_bytes
    used = 2 * sum(seq_lengths) * n_heads * d_head * dtype_bytes
    return {
        'allocated_bytes': allocated,
        'used_bytes': used,
        'waste_frac': (allocated - used) / allocated,
    }

def paged_memory(seq_lengths: list[int], block_size: int, d_head: int, n_heads: int, dtype_bytes: int = 2) -> dict:
    """
    Paged KV cache: each sequence needs ceil(length / block_size) blocks.
    Waste is the unused offsets in each sequence's last block.
    Returns {'blocks_allocated', 'blocks_used_frac', 'allocated_bytes', 'used_bytes', 'waste_frac'}.
    """
    ...
```

Run:
```python
# Realistic scenario: variable-length prompts in a batch of 32
np.random.seed(0)
lengths = np.random.randint(50, 2048, size=32).tolist()

c = contiguous_memory(lengths, d_head=128, n_heads=32)
p = paged_memory(lengths, block_size=16, d_head=128, n_heads=32)

print(f"contiguous:  {c['allocated_bytes']/1e9:.2f} GB allocated, {c['waste_frac']*100:.1f}% waste")
print(f"paged (16):  {p['allocated_bytes']/1e9:.2f} GB allocated, {p['waste_frac']*100:.1f}% waste")
print(f"savings:     {(1 - p['allocated_bytes']/c['allocated_bytes'])*100:.1f}%")
```

**Expected**: contiguous wastes 40-60% depending on length distribution. Paged wastes < `block_size / min(lengths)` — usually 1-5%. Savings 40-60%.

**Bigger `block_size`** = less bookkeeping but more per-block waste. Smaller block_size = less waste but more allocator overhead + more block-table lookups.

## The 3 lessons, memorized

1. **PagedAttention allocates KV in fixed-size blocks, not contiguous per-sequence buffers.** Block size is typically 16 tokens.
2. **Each sequence has a block table** mapping logical position → physical block. Attention gathers via the table.
3. **Padding waste in variable-length batches drops from ~40% to ~5%.** Frees ~40% of KV cache memory for larger batches or longer contexts.

## What to say aloud (~90 seconds)

> "Contiguous KV cache allocates (B, T_max, ...) per batch. In production, request lengths vary wildly — you end up wasting 40-60% of KV memory on padding.
>
> PagedAttention takes the OS-virtual-memory idea: allocate KV cache in fixed-size blocks, typically 16 tokens each. Each sequence has a block table — a list of block indices pointing into a shared physical pool. When a sequence grows past its current block, it grabs a new one from a free pool. When it finishes, its blocks return to the pool.
>
> The attention kernel now gathers K and V through the block table instead of a contiguous read. Slight overhead but eliminates padding waste. In practice, memory usage drops ~40%, letting you either serve larger batches (higher throughput) or longer contexts. Also enables continuous batching — new sequences can join a running batch by claiming free blocks."

## References (after timer)

- vLLM blog post: "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention" — the introductory writeup
- Kwon et al. 2023, "Efficient Memory Management for Large Language Model Serving with PagedAttention" — the paper

## Attempt header

```python
# Attempt N — YYYY-MM-DD
# Exercises passing: 1/3, 2/3, 3/3
# Notes:
```
