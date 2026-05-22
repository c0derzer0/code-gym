# Transformer block

## Problem

Full pre-norm transformer decoder block: LayerNorm → MHA → residual → LayerNorm → FFN → residual. No reuse of existing transformer modules.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        # default d_ff = 4 * d_model
        ...
    def forward(self, x):
        """x: (B, T, d_model) → (B, T, d_model)"""
        ...
```

## Things to get right

- **Pre-norm** (LN before sub-layer), not post-norm. Modern LLMs use pre-norm — more stable training.
- FFN: `Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)`. `d_ff = 4 * d_model` is the standard.
- Residual connections wrap each sub-layer.
- Dropout typically applied after attention output and inside the FFN.

## Be ready to discuss

- Pre-norm vs post-norm — why pre-norm wins.
- Why `d_ff = 4 * d_model` (mostly empirical, but FFN dominates parameter count).
- GELU vs ReLU vs SwiGLU.

## Test

- Shape preservation: `(B, T, d_model)` → same.
- Gradient flows end-to-end (`loss.backward()` succeeds, no NaN).
- Param count sanity-check against a known config (e.g., GPT-2 small block ≈ 7M params for d=768, h=12).

## References (after timer)

- nanoGPT `Block`
- "On Layer Normalization in the Transformer Architecture" (pre-norm paper)
