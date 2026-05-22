# Adam optimizer

## Problem

Adam from scratch as a custom optimizer, including bias correction.

```python
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        ...
    def zero_grad(self): ...
    def step(self): ...
```

## Things to get right

- Per-param buffers: `m` (first moment), `v` (second moment), both zero-init.
- Step counter `t` increments each `step()`.
- Update:
  - `m = beta1 * m + (1 - beta1) * grad`
  - `v = beta2 * v + (1 - beta2) * grad^2`
  - `m_hat = m / (1 - beta1^t)`
  - `v_hat = v / (1 - beta2^t)`
  - `p = p - lr * m_hat / (sqrt(v_hat) + eps)`
- `eps` is *outside* the sqrt in PyTorch's Adam, *inside* in the original paper. Match PyTorch.

## Test

- Train a tiny MLP for 100 steps with your Adam; compare to `torch.optim.Adam` — final params within `1e-5`.
- Forget bias correction → losses diverge from PyTorch's. Good way to confirm you implemented it.
