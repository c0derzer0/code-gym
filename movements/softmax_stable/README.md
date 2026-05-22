# Stable softmax

## Problem

Numerically stable softmax from scratch in numpy (and verify in PyTorch).

```python
def softmax(x, axis=-1):
    """Stable softmax along the given axis."""
    ...
```

## Things to get right

- Subtract max along axis *before* exp. Otherwise large logits → `exp(1000) = inf`.
- Keep `keepdims=True` when reducing for broadcastable subtraction.

## Test

- `softmax([1, 2, 3])` ≈ `[0.0900, 0.2447, 0.6652]`.
- `softmax([1000, 1001, 1002])` returns the same result — *no NaN*. (This is the test.)
- Sum along axis = 1.0 exactly (within float precision).
- Compare against `scipy.special.softmax` or `torch.softmax`.
