# Backprop from scratch

Foundational movement. Build a 2-layer MLP binary classifier in pure numpy — forward pass, backward pass (chain rule by hand), SGD update, and a training loop that visibly converges.

The pieces tested here — manual gradient bookkeeping, per-layer `dL/dW` and `dL/dx`, the sigmoid+BCE combined-gradient shortcut, a working SGD step — are the substrate every higher-level ML system is built on. If you can write this cold in 45 minutes you can write almost any small from-scratch training loop.

## Problem

Numpy only — no PyTorch, no autograd, no torch operations.

Binary classification on synthetic data. Generate `(N, in_dim)` features, label `y = 1 if sum(x) > 0 else 0`. Train a 2-layer MLP (hidden_dim → 1) with sigmoid activations and BCE loss until accuracy > 90%.

```python
class Linear:
    def __init__(self, in_dim, out_dim): ...
    def forward(self, x):  ...  # returns (N, out_dim), caches x
    def backward(self, dL_dout):  ...  # computes dL/dW, dL/db, returns dL/dx

class NN:
    def __init__(self, in_dim, hidden_dim, out_dim): ...
    def forward(self, x):  ...  # returns probability (N, 1)
    def backward(self, y):  ...  # propagates gradients through both layers
    def step(self, lr):  ...   # SGD update

def bce_loss(out, y):  ...  # scalar mean loss
```

## What the test should show

```
epoch   0 | loss 0.69 | acc 0.50
epoch  50 | loss 0.30 | acc 0.85
epoch 100 | loss 0.15 | acc 0.93
epoch 200 | loss 0.05 | acc 0.97
```

Loss drops monotonically. Accuracy climbs to >90% within 200 epochs on the toy task.

## Things to get right

### Forward pass — cache for backward

Every layer needs to remember its input. `Linear.forward(x)` stores `self.x = x` so `Linear.backward` can use it.

### Backward pass — three quantities per layer

For `Linear` with input `x: (N, in_dim)`, weight `W: (in_dim, out_dim)`, output `out: (N, out_dim)`:
- `dL/dW = x.T @ dL_dout`              shape `(in_dim, out_dim)`
- `dL/db = dL_dout.sum(axis=0)`        shape `(out_dim,)`
- `dL/dx = dL_dout @ W.T`              shape `(N, in_dim)` — return upstream

### Sigmoid backward

Cache `sig = σ(z)` from forward. Then:
- `dL/dz = dL/dσ · σ · (1 − σ)`

The derivative is evaluated AT the cached `σ`, NOT at the gradient flowing back. Common bug.

### Sigmoid + BCE — the combined gradient shortcut

When BCE is computed on a sigmoid output, the gradient w.r.t. the **logit** (pre-sigmoid) simplifies to:
```
dL/dz = (σ(z) − y) / N
```

No need to compute `dL/dσ = -(y/σ - (1-y)/(1-σ))/N` and chain through sigmoid separately. The algebra cancels. Use this in `NN.backward` — start backprop at the logit, skip the per-layer sigmoid backward on the output.

### BCE numerical stability

`y · log(σ) + (1−y) · log(1−σ)` blows up if σ hits 0 or 1. Clip:
```python
out = np.clip(out, 1e-12, 1 - 1e-12)
```

### Training loop

```
for epoch in range(N_EPOCHS):
    out = model.forward(X)
    loss = bce_loss(out, y)
    model.backward(y)         # computes gradients, stores in each layer
    model.step(lr)            # applies SGD update
```

Mini-batch optional for this problem — full-batch SGD converges in <200 epochs on the toy data.

## Synthetic data spec

```python
np.random.seed(0)
N, in_dim, hidden_dim = 200, 4, 8
X = np.random.randn(N, in_dim)
y = (X.sum(axis=1, keepdims=True) > 0).astype(float)   # (N, 1) — 0 or 1, NOT Gaussian
```

Critical: `y` must be `{0, 1}` shaped `(N, 1)` to match `out`. NOT `np.random.randn` (that gives real-valued labels — BCE goes NaN).

## Common bugs

1. **Sigmoid operator precedence**: `1 / 1 + np.exp(-x)` is `1 + exp(-x)`. Need parens: `1 / (1 + np.exp(-x))`.
2. **`y` not in {0, 1}**: random.randn gives reals; BCE NaNs. Use `> threshold` and cast to float.
3. **Shape mismatch on `y`**: `(N,)` vs `(N, 1)`. Broadcasting issues silently.
4. **`Linear.backward` doesn't cache `dL/dW`** — gradient computed but never stored, optimizer step does nothing.
5. **No `dL/dx` return** — upstream layer gets nothing to chain from.
6. **Sigmoid backward evaluated at the wrong thing** — must use the cached forward sigmoid, not the gradient flowing back.
7. **Missing `lr` in the update step** — `W -= dW` instead of `W -= lr * dW`. Big steps blow up training.
8. **Mean vs sum in BCE** — be consistent. `(out - y) / N` is the mean-loss gradient; `(out - y)` is the sum-loss gradient.
9. **Bias broadcasting**: `b` is `(out_dim,)`, `x @ W` is `(N, out_dim)` — adds broadcast. `dL/db` is `dL_dout.sum(axis=0)`, not `dL_dout.mean(...)`.

## References (after timer)

- Karpathy's [micrograd](https://github.com/karpathy/micrograd) — scalar-level backprop, ~150 lines.
- Karpathy's "Neural Networks: Zero to Hero" video 1.
- Stanford CS231n notes on backpropagation.
- "Matrix calculus for backprop" cheat sheet (Justin Johnson).

## Walkthrough flow (for explaining the algorithm aloud)

The narrative that makes this implementation legible to a reader:

1. **Frame the problem**: binary classifier, single sigmoid output, BCE loss, pure numpy.
2. **Sketch forward**: two linears, sigmoid on hidden, sigmoid on output, BCE.
3. **Sketch backprop**: each linear caches its input. Backward computes `dL/dW = x.T @ dL_dout`, `dL/dx = dL_dout @ W.T`, returns upstream. For sigmoid+BCE the combined gradient simplifies to `(p - y) / N` w.r.t. the logit.
4. **Code top-down**: Linear class → NN class → loss → training loop.
5. **Test on synthetic data, watch loss drop.** That's the closer.

## Variants for future attempts

Once the binary classifier lands cleanly under 45 min:

- **Multi-class** — replace sigmoid with softmax, BCE with cross-entropy. The `(p - y) / N` gradient generalizes — `(softmax(z) - one_hot(y)) / N`.
- **Regression** — linear output, MSE loss. Different gradient at the head.
- **ReLU activation** — replace sigmoid in hidden layer. Gradient: `1 if z > 0 else 0` (cached from forward).
- **Mini-batch SGD** — shuffle and slice in the training loop. Same model, different loop.
- **Adam optimizer** — replace `step()` with your Adam from movement `adam/`. Reuse the same Linear class.

Each is a 15-30 min variant. Stack them — by the end you have a fluent "neural net from scratch" toolkit.

## Attempt header template

```python
# Attempt N — YYYY-MM-DD
# Started:
# First pass (loss drops, no NaN):
# Correct (acc > 90% on toy data):
# Stuck on:
# Notes:
```
