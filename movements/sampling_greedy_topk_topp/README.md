# Sampling — greedy, temperature, top-k, top-p

Four sampling functions used in every LLM inference path. All take `logits: (V,)` and return a sampled token id. Muscle memory movement.

Total: ~60 min.

## Signatures

```python
def greedy_sample(logits: torch.Tensor) -> int:
    """
    logits: (V,) — vocab-sized
    Returns: argmax token id.
    """

def temperature_sample(logits: torch.Tensor, temperature: float) -> int:
    """
    Divide logits by temperature, softmax, sample.
    T = 1 → normal. T < 1 → peakier. T > 1 → flatter. T → 0 → greedy.
    """

def top_k_sample(logits: torch.Tensor, k: int, temperature: float = 1.0) -> int:
    """
    Keep top-k logits, set rest to -inf. Apply temperature. Softmax. Sample.
    """

def top_p_sample(logits: torch.Tensor, p: float, temperature: float = 1.0) -> int:
    """
    Sort logits descending. Keep smallest prefix whose CDF >= p.
    Set rest to -inf. Apply temperature. Softmax. Sample.
    """
```

## Tests

```python
torch.manual_seed(0)
V = 10
logits = torch.randn(V)

# greedy = argmax
assert greedy_sample(logits) == int(logits.argmax())

# temperature=0 → greedy (approximate with tiny value)
assert temperature_sample(logits, temperature=0.001) == int(logits.argmax())

# top-k=1 → greedy
assert top_k_sample(logits, k=1) == int(logits.argmax())

# top-p sampling with p=0.001 → picks the single largest
assert top_p_sample(logits, p=0.001) == int(logits.argmax())

# Statistical check: temperature=1.0 sampling should match torch.multinomial distribution
counts = torch.zeros(V)
n_samples = 10000
for _ in range(n_samples):
    counts[temperature_sample(logits, temperature=1.0)] += 1
empirical = counts / n_samples
expected = torch.softmax(logits, dim=-1)
assert torch.allclose(empirical, expected, atol=0.02)
```

## Watch out for

- **`torch.multinomial(probs, num_samples=1)`** samples from a distribution. Docs say probs must be non-negative.
- **Top-p**: cumulative sum on SORTED logits. Use `torch.cumsum(probs_sorted, dim=-1)`. Find first index where `cumsum >= p`. Keep everything up to and including that index.
- **Filter after sort, unsort back**: easiest to mask on the sorted order, then scatter back.
- **Numerical stability**: apply softmax to `(logits - max) / T` — don't forget the max subtract.
- **`k > V`**: cap at V.
- **`p >= 1.0`**: keep everything.

## What to say aloud

> "Greedy = argmax. Temperature = divide logits by T before softmax — lower T = more deterministic. Top-k = keep the k largest logits, mask rest with -inf. Top-p (nucleus) = sort logits, find smallest prefix whose cumulative probability ≥ p, keep only those. In production these are stacked: sample = top-p (top-k (temperature (logits))). Usually top-k first (fast filter to a small pool), then top-p, then temperature."
