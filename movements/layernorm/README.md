# LayerNorm

## Problem

LayerNorm from scratch as an `nn.Module`. Include learnable affine parameters (gamma, beta).

```python
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        ...
    def forward(self, x):
        """Normalize over the last dim."""
        ...
```

## Things to get right

- Normalize over the last dim (or `normalized_shape`), not batch dim. Computes mean/var **per sample**.
- `gamma` (scale) initialized to ones, `beta` (shift) to zeros.
- `eps` added inside the sqrt for numerical stability: `(x - mean) / sqrt(var + eps)`.
- `var = mean((x - mean)^2)`, biased (no Bessel correction). Matches `F.layer_norm`.

## Test

- Shape `(B, T, D)` → same shape out.
- After normalization (with default affine), mean ≈ 0, var ≈ 1 along last dim.
- Compare to `F.layer_norm(x, (D,))` within `1e-5`.
- Gradient flows through `gamma` and `beta`.

## Be ready to discuss

- LayerNorm vs BatchNorm — when each is used and why. LN doesn't depend on batch stats → friendlier to variable-length sequences and small batches.
