# Attempt 1 — May 26, 2026
# Started: 9:21pm
# First pass (runs): 9:41pm (~20 min)
# Finished with hints and corrections: 10:00pm (~40 min total)
# Correct (matches F.layer_norm within 1e-5): yes
# Stuck on: gamma/beta declaration (used register_buffer first), variance correction
# Notes (bugs caught + fixes, with hints):
#   - gamma/beta were declared as buffers — wrong. they're LEARNABLE params, need
#     gradients. fix: nn.Parameter (init gamma=ones, beta=zeros). buffer is for
#     persistent state with no grad (e.g., causal mask in MHA).
#   - used torch.var with default unbiased=True (Bessel correction). F.layer_norm
#     uses BIASED variance. fix: ((x - mean)**2).mean(dim=-1, keepdim=True). gave
#     a 1/sqrt(2) factor on output for D=2; expected ±1, was getting ±0.7071.
#   - confused dim=normalized_size vs dim=full_input_shape. fixed: pass just the
#     last-dim size (e.g., LayerNorm(D) for input (B, T, D)), since gamma/beta
#     are per-feature and broadcast over (B, T).
#   - tested with D=2 (input (2,2)) — output is always ±1 by construction. switch
#     to D=8+ for visual variety in the comparison.
# Concept locked: gamma/beta are PER-FEATURE (shape (D,)) and SHARED across batch
# and positions. mean/var are per-sample/position (shape (B, T, 1)). LN flips
# BN's axes — normalizes within a sample across features, not across batch.

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):

        mean = torch.mean(x, dim=-1, keepdim=True)
        var = ((x - mean)**2).mean(dim=-1, keepdim=True)
        #print(mean.shape, var.shape, self.gamma.shape, self.beta.shape)

        ln = ((x - mean) / torch.sqrt(var + self.eps)) * self.gamma + self.beta

        #print(ln)

        return ln

in_shape = (7, 28)
x = torch.randn(in_shape)
#print(x)
ln = LayerNorm(in_shape[-1])
out = ln(x)
ref = torch.nn.functional.layer_norm(x, (in_shape[-1],))

print(torch.allclose(out, ref, atol=1e-5))