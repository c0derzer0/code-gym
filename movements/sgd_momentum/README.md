# SGD with momentum

## Problem

SGD with momentum from scratch as a custom optimizer. No `torch.optim`.

```python
class SGDMomentum:
    def __init__(self, params, lr, momentum=0.9):
        ...
    def zero_grad(self): ...
    def step(self): ...
```

## Things to get right

- Velocity buffer per-param, initialized to zeros.
- Update rule (PyTorch convention): `v = momentum * v + grad`, then `p = p - lr * v`. Other frameworks differ — pick one and stick with it.
- Use `with torch.no_grad():` inside `step` when modifying `p.data`. Or update `p.data` directly.
- `zero_grad`: set `p.grad` to None or zero in-place.

## Test

- Train a tiny MLP on synthetic regression for 50 steps with your optimizer; train an identical net with `torch.optim.SGD(lr=..., momentum=...)`. Final losses should match within `1e-4`.
