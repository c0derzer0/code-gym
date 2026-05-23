# Triangular (causal) mask

Also known as: lower-triangular mask, causal mask, autoregressive mask. The mask used inside causal self-attention so token `t` can't attend to tokens `> t`.

## Problem

Implement two variants from scratch in PyTorch — no `torch.tril`, no `torch.triu`.

### 1. Boolean mask

```python
def causal_mask_bool(seq_len: int) -> torch.Tensor:
    """
    Returns a (seq_len, seq_len) bool tensor.
    mask[i, j] = True  iff j <= i   (token i may attend to token j)
    """
    ...
```

### 2. Additive mask (for use inside attention before softmax)

```python
def causal_mask_additive(seq_len: int) -> torch.Tensor:
    """
    Returns a (seq_len, seq_len) float tensor.
    mask[i, j] = 0.0         iff j <= i
    mask[i, j] = -inf        iff j > i
    Adding this to attention logits before softmax zeroes future positions.
    """
    ...
```

## Things to get right

- Build it from primitives: `torch.arange`, broadcasting comparison `i >= j`. Don't use `torch.tril`.
- The right way: `i = torch.arange(T)[:, None]`, `j = torch.arange(T)[None, :]`, then `i >= j`.
- For additive: start from bool, then `mask.float().masked_fill(~bool_mask, float('-inf'))` (or build `where(bool_mask, 0.0, -inf)`).
- Inside an `nn.Module`, register the mask as a **buffer** (`self.register_buffer('mask', ...)`), not a parameter — it moves with `.to(device)` but has no gradient.
- Cache by `max_seq_len`, slice `[:T, :T]` at use time. Don't rebuild per forward.

## Test

- `causal_mask_bool(4)` →
  ```
  [[ T, F, F, F],
   [ T, T, F, F],
   [ T, T, T, F],
   [ T, T, T, T]]
  ```
- `causal_mask_additive(3)` →
  ```
  [[ 0, -inf, -inf],
   [ 0,    0, -inf],
   [ 0,    0,    0]]
  ```
- Verify against `torch.tril(torch.ones(T, T)).bool()` — must match exactly.
- Sanity: `softmax(attn_scores + causal_mask_additive(T))` zeros out future positions.

## Be ready to discuss

- Why -inf and not 0? Softmax rescales — adding a large negative number is the trick that zeros it post-softmax. Adding 0 changes nothing.
- Why a buffer, not a parameter? No gradient, but needs to follow `.cuda()/.to()`.
- Bidirectional (BERT) attention uses no mask. Encoder-decoder uses a different cross-mask. Sliding window (Mistral) bands the mask.
