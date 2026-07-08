# Grouped Query Attention (GQA)

Modern LLM standard (Llama 3, Mistral, Gemma). Small variant of MHA where `n_kv_heads < n_q_heads`: multiple query heads share the same key/value head. Cuts KV cache memory by the grouping factor.

## Problem

Extend your MHA to support GQA. Q keeps its usual number of heads. K and V produce fewer heads and get **broadcast** across groups of Q heads before the attention math.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, max_seq_len, d_model, n_q_heads, n_kv_heads, causal=True):
        """
        n_q_heads = 32, n_kv_heads = 8  → each K/V head shared across 4 Q heads
        n_kv_heads must divide n_q_heads
        """
        ...

    def forward(self, x):
        """
        x: (B, T, d_model)
        Returns: (B, T, d_model)
        """
        ...
```

## The math

`d_head = d_model // n_q_heads` (same as before — d_head is defined by Q).

Projections:
- `q_proj`: `d_model → n_q_heads · d_head` = `d_model → d_model` (unchanged from MHA).
- `k_proj`: `d_model → n_kv_heads · d_head` (**smaller**).
- `v_proj`: `d_model → n_kv_heads · d_head` (**smaller**).
- `out_proj`: `d_model → d_model` (unchanged).

After the projections and head reshape:
- `q`: `(B, n_q_heads, T, d_head)`
- `k`: `(B, n_kv_heads, T, d_head)`
- `v`: `(B, n_kv_heads, T, d_head)`

**Broadcast K, V across Q head groups** before attention. Two ways:

```python
group_size = n_q_heads // n_kv_heads

# Option A — repeat_interleave
k = k.repeat_interleave(group_size, dim=1)   # → (B, n_q_heads, T, d_head)
v = v.repeat_interleave(group_size, dim=1)   # → (B, n_q_heads, T, d_head)

# Option B — expand (no copy — memory efficient)
# reshape to add a group axis, then expand
k = k[:, :, None, :, :].expand(-1, -1, group_size, -1, -1).reshape(B, n_q_heads, T, d_head)
```

Option A is simpler. Option B is more memory-efficient (a view; no data copy). Production impls use Option B.

Then attention proceeds normally: `Q @ K^T / √d_head`, causal mask, softmax, `@ V`, reshape, out_proj.

## Memory savings

KV cache size scales with `n_kv_heads`, not `n_q_heads`.

For Llama-3-70B: `n_q_heads = 64, n_kv_heads = 8` → **8× smaller KV cache** vs full MHA.

For a T=4096 context, this drops KV cache from ~10 GB/sequence (full MHA) to ~1.3 GB/sequence. **Load-bearing memory saving.**

## Test

```python
torch.manual_seed(0)
B, T, d_model, n_q_heads, n_kv_heads = 2, 8, 64, 8, 2
gqa = GroupedQueryAttention(max_seq_len=8, d_model=d_model, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads)

x = torch.randn(B, T, d_model)
out = gqa(x)

# Shape check
assert out.shape == x.shape

# Causality property (same test as your MHA):
# perturbing x[:, t, :] must NOT change out[:, :t, :]
x_perturbed = x.clone()
x_perturbed[:, 3, :] += 100
out_perturbed = gqa(x_perturbed)
assert torch.allclose(out[:, :3, :], out_perturbed[:, :3, :], atol=1e-5)
print("✓ shape preserved + causality holds")

# Parameter count sanity: should be less than a full-MHA equivalent
param_count = sum(p.numel() for p in gqa.parameters())
mha_equivalent = 4 * d_model * d_model + 4 * d_model  # rough — 4 linears
print(f"GQA params: {param_count}, full-MHA equivalent: ~{mha_equivalent}")
# GQA should be smaller — K, V projections are (n_kv_heads/n_q_heads)× smaller
```

## Things to get right

- `n_kv_heads` must divide `n_q_heads`. Assert it.
- **`d_head` is defined by `n_q_heads`, not `n_kv_heads`.** `d_head = d_model // n_q_heads`.
- K/V projections output `n_kv_heads · d_head`, NOT `d_model`.
- Broadcast (repeat_interleave or expand) K, V along the head dim BEFORE the attention matmul.
- Same causal mask logic as your MHA — attention shape is `(B, n_q_heads, T, T)`.

## Common bugs

1. **Computing `d_head = d_model // n_kv_heads`**: wrong. `d_head` is per-Q-head. Uses `n_q_heads`.
2. **Making K/V projections output `d_model`**: no memory savings. Point of GQA is that K, V are smaller.
3. **Broadcasting K, V at the wrong dim**: should be along the head dim (`dim=1` after transpose), not batch or seq.
4. **Forgetting the divisibility check**: `n_kv_heads = 3, n_q_heads = 8` will produce garbage silently. Assert.
5. **Using `n_kv_heads` in the attention scale**: no, scale is `1 / √d_head`, same as MHA.

## What to say aloud

- Llama 3 70B has 64 Q heads and 8 KV heads → GQA-8. Each KV head is shared across 8 Q heads.
- Empirically, GQA has ~0% quality drop compared to full MHA. Sometimes even slightly better (regularization effect).
- Multi-Query Attention (MQA) is the extreme: `n_kv_heads = 1`. Used in some models but too aggressive for most.
- The savings only matter for the KV cache during inference. Training memory savings are marginal.

## References (after timer)

- GQA paper (Ainslie et al., 2023) — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
- Llama 3 model card — mentions their exact `(n_q_heads, n_kv_heads)` per model size

## Attempt header template

```python
# Attempt N — YYYY-MM-DD
# Started:
# First pass (runs without error):
# Correct (shape + causality holds):
# Stuck on:
# Notes:
```
