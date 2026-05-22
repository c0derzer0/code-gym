# RMSNorm

## Problem

Root-mean-square LayerNorm — what Llama/Mistral/Gemma use instead of LayerNorm.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        ...
    def forward(self, x):
        """Normalize by RMS along last dim, then scale by gamma."""
        ...
```

## Things to get right

- `rms(x) = sqrt(mean(x^2) + eps)`. **No mean subtraction.** That's the difference from LayerNorm.
- Only one learnable parameter: `gamma` (scale). No `beta` (shift).
- `out = x / rms(x) * gamma`.

## Test

- Shape preservation.
- For random `x`, manually compute RMS along last dim, divide, multiply by `gamma=1` — match output.
- Compare to `nn.RMSNorm` (PyTorch ≥ 2.4) if available.

## Be ready to discuss

- Why RMSNorm > LayerNorm in modern LLMs: ~2x faster, comparable quality, and the mean-subtraction in LN turns out not to matter much. Validated empirically.
