# Flash Attention v2 (full)

Full Flash Attention 2 in numpy. Tiled Q, tiled K/V, online softmax within each tile pair, output accumulator with rescale. Not a Triton implementation — a numerically-verified reference impl that makes the paper trivially readable.

Total: **2-3 hours.** Prerequisite: `online_softmax_recurrence` done.

## What you'll build

Given `Q, K, V` of shape `(B, H, T, d_head)` — same as your MHA — compute attention output using the FA-2 algorithm:

- Never materialize the full `T × T` attention matrix
- Tile Q into blocks of size `Bc` (typically 128)
- Tile K/V into blocks of size `Br` (typically 128)
- For each Q tile: loop over K/V tiles, maintaining running softmax state per Q row
- Output accumulator is rescaled every time the running max updates
- After all K/V tiles: normalize by final denominator

## The algorithm (FA-2)

```
for i in range(0, T, Bc):                                # outer loop: Q tiles
    Q_i = Q[..., i:i+Bc, :]                              # (B, H, Bc, d)
    O_i = zeros(B, H, Bc, d)                             # output accumulator
    m_i = -inf * ones(B, H, Bc, 1)                       # running max per Q row
    l_i = zeros(B, H, Bc, 1)                             # running denominator
    
    for j in range(0, T, Br):                            # inner loop: K/V tiles
        K_j = K[..., j:j+Br, :]                          # (B, H, Br, d)
        V_j = V[..., j:j+Br, :]                          # (B, H, Br, d)
        
        S_ij = Q_i @ K_j.transpose(-2, -1) / sqrt(d)     # (B, H, Bc, Br)
        # Apply causal mask if j > i (or slice within-tile diagonal)
        
        m_ij = max(S_ij, dim=-1, keepdim=True)           # per-Q-row max of this tile
        m_new = max(m_i, m_ij)
        P_ij = exp(S_ij - m_new)                         # softmax numerator
        l_ij = sum(P_ij, dim=-1, keepdim=True)
        
        # Rescale old accumulator and denominator by exp(m_i - m_new)
        alpha = exp(m_i - m_new)
        O_i = O_i * alpha + P_ij @ V_j
        l_i = l_i * alpha + l_ij
        m_i = m_new
    
    O_i = O_i / l_i                                       # final normalization
    O[..., i:i+Bc, :] = O_i
```

## Signature

```python
def flash_attention_v2(Q, K, V, block_size_q=128, block_size_kv=128, causal=True):
    """
    Q, K, V: (B, H, T, d_head), numpy arrays or torch tensors
    Returns: (B, H, T, d_head) — same as softmax(QK^T/√d) @ V
    """
    ...
```

Also implement:

```python
def vanilla_attention(Q, K, V, causal=True):
    """Reference: materialize full T×T matrix."""
    ...
```

## Tests

```python
np.random.seed(0)
B, H, T, d = 2, 4, 64, 32
Q = np.random.randn(B, H, T, d)
K = np.random.randn(B, H, T, d)
V = np.random.randn(B, H, T, d)

out_flash = flash_attention_v2(Q, K, V, block_size_q=16, block_size_kv=16)
out_vanilla = vanilla_attention(Q, K, V)

assert out_flash.shape == out_vanilla.shape
assert np.allclose(out_flash, out_vanilla, atol=1e-5)
print("✓ FA-2 matches vanilla attention")

# Test with various block sizes
for bs in [4, 8, 16, 32, 64]:
    out = flash_attention_v2(Q, K, V, block_size_q=bs, block_size_kv=bs)
    assert np.allclose(out, out_vanilla, atol=1e-5), f"failed at block_size={bs}"
print("✓ correct across all block sizes")
```

## HBM traffic comparison

```python
def count_flash_hbm(B, H, T, d, block_size_q, block_size_kv, dtype_bytes=2):
    """
    Flash Attention HBM reads:
      - Q loaded T/Bc times... actually once per outer iter, so T/Bc × Bc × d = T × d per head
      - K, V loaded T/Bc times each (once per Q tile) but Wait: FA-2 iterates Q inner, K outer. Re-check.
      - Output written T × d per head
    """
    ...

def count_vanilla_hbm(B, H, T, d, dtype_bytes=2):
    """
    Vanilla attention HBM:
      - Read Q, K, V (T × d each)
      - Write attention matrix (T × T)
      - Read attention matrix
      - Write output (T × d)
    """
    ...
```

Show: FA-2 HBM is `O(T · d)`; vanilla is `O(T²)`. For `T=8192, d=128`: FA-2 = 1M elements per head, vanilla = 67M elements per head. **67× less HBM traffic.**

## Watch out for

1. **Causal masking with tiles**: when Q tile `i` and KV tile `j` overlap (some Q positions before some K positions), you need to apply the causal mask WITHIN the tile.
2. **Rescale order**: `O_i = O_i * alpha + P_ij @ V_j`. If you compute `P_ij @ V_j` first and add to unrescaled `O_i`, you're mixing summands at different scales — wrong result.
3. **Initial state**: `m_i = -inf`, `l_i = 0`, `O_i = 0`. First iteration: `alpha = exp(-inf - m_ij_first) = 0`, so old accumulator contributes nothing (correct).
4. **Numerical stability**: subtracting `m_new` before exp keeps values bounded.
5. **The FA-2 vs FA-1 difference**: FA-1 loops K outer, Q inner. FA-2 loops Q outer, K inner. The Q-outer choice better utilizes GPU parallelism — different tiles of Q are independent, so you can parallelize across them.

## What to say aloud (~2 min)

> "Flash Attention 2 tiles Q into outer blocks of size Bc, K and V into inner blocks of size Br. For each Q tile, we loop over all K/V tiles maintaining a running softmax state per Q row: running max m, running denominator l, and running output accumulator O. When a new K tile arrives, we compute the partial attention scores, take the row-max, and if that new max is larger than the running max, we rescale the old output and denominator by exp(m_old - m_new). Then we add the new tile's contribution.
>
> After all K/V tiles: divide O by l for the final normalized output. The full T×T attention matrix is never materialized in HBM — only tile pairs live in on-chip SRAM at any time. HBM traffic collapses from O(T²) to O(T·d).
>
> On A100/H100, this transforms attention from a bandwidth-bound to a compute-bound kernel. FA-2 improved on FA-1 by swapping the loop order — Q-outer parallelizes across Q tiles, which better utilizes the SM. Empirically 2× faster than FA-1 on modern GPUs, 5-9× vs vanilla PyTorch attention."

## Followups

- Add a benchmark: time vanilla vs FA-2 for various T. Show wall-clock reflects HBM traffic ratio (in numpy, both are CPU-bound so the ratio won't match GPU exactly — but the structural point holds).
- Implement backward pass: FA-2 backward recomputes the softmax from stored (m, l) rather than storing the full attention matrix.
- Port to Triton for real GPU speedup (a much bigger project).

## References

- Dao 2023, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
- Dao et al. 2022, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- The Triton FA-2 tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
