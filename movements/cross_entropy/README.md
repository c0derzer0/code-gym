# Cross-entropy loss

## Problem

Cross-entropy from logits, numerically stable, in numpy. Then verify against PyTorch.

```python
def cross_entropy(logits, targets):
    """
    logits: (N, C)
    targets: (N,) int class indices
    Returns: scalar mean loss.
    """
    ...
```

## Things to get right

- Use log-sum-exp form: `loss = -logits[targets] + logsumexp(logits)`. Don't compute softmax-then-log — that loses precision.
- `logsumexp(x) = max(x) + log(sum(exp(x - max(x))))`.
- Mean reduction by default (matches `F.cross_entropy`).

## Test

- Random `logits` of shape `(32, 10)`, random integer `targets` in `[0, 10)` → compare to `F.cross_entropy(logits, targets)` within `1e-6`.
- Verify gradient: `d L / d logits` = `softmax(logits) - one_hot(targets)`, divided by N.
