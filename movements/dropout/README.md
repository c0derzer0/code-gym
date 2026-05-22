# Dropout

## Problem

Dropout from scratch as an `nn.Module` with correct train/eval behavior.

```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        ...
    def forward(self, x):
        """At train: zero each element w.p. p, scale survivors by 1/(1-p).
           At eval: identity."""
        ...
```

## Things to get right

- **Inverted dropout**: scale survivors by `1/(1-p)` during training, identity at inference. This is what every framework does so inference is a no-op.
- Use `self.training` (inherited from `nn.Module`) to gate behavior.
- Mask must be sampled fresh per forward pass.
- Same mask broadcast across the elements being dropped — sample with the input shape.

## Test

- `module.train()`: with `p=0.5`, output has ~50% zeros, non-zero values are 2x input.
- `module.eval()`: output exactly equals input.
- Expectation: `E[dropout(x)] = x` during training (this is what inverted dropout buys you).
- Variance increases — sanity-check `var(dropout(x)) > var(x)` for non-trivial input.
