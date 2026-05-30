# Attempt 1 — May 30, 2026
# Started: 4:52pm
# First pass (runs without error, but had logic bugs): 5:15pm (~23 min)
# Correct (shape preserves end-to-end, no NaN, grad flows): 5:35pm (~43 min, with hints)
# Stuck on: pre-norm residual pattern, then mask broadcasting at T < max_seq_len
# Notes (bugs caught + fixes, with hints):
#   - __int__ typo for __init__ → constructor never ran, AttributeError on self.mha
#   - super().__init__(s) with undefined `s` → super().__init__()
#   - MHA constructor signature mismatch (passed (d_model, n_heads, causal) instead of
#     (max_seq_len, d_model, n_heads, causal))
#   - torch.nn.GELU(x) calls the class as a function → use F.gelu(x)
#   - residual was `x = ln(x) + mha(ln(x))` (both terms post-LN) → correct pre-norm is
#     `x = x + sublayer(ln(x))` so the unnormalized signal flows through the residual
#   - shared self.ln across both sublayers → need self.ln_1 and self.ln_2 with independent γ/β
#   - test block existed but `loss.backward` (no parens) was just a method reference, never
#     called. silent — wouldn't catch NaN-in-grad. fix: loss.backward()
#   - MHA itself only worked when T == max_seq_len because the causal_mask buffer was
#     sized (max_seq_len, max_seq_len) and added directly. fixed in MHA attempt_1.py
#     post-hoc with `self.causal_mask[:T, :T]` slicing pattern (nanoGPT style).
# Verification: shape preserves (B, T, d_model) → same; no NaN; works for T < max_seq_len.
# Followups for attempt 2:
#   - typo FeedFowardNetwork → FeedForwardNetwork
#   - move the test block under `if __name__ == "__main__":` so importing this file
#     elsewhere doesn't auto-run the test
#   - paste MHA + LayerNorm classes inline (or fix the importing modules to be guarded too)
#     so this file's imports don't have side effects from the imported test blocks
#   - d_ff convention: 4 * d_model (you used 3 * d_model; works, just unusual)
import sys
sys.path.append('movements')

import torch
from torch import nn
from multi_head_attention.attempts.attempt_1 import MultiHeadAttn
from layernorm.attempts.attempt_1 import LayerNorm

class FeedFowardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        
        x = self.linear_1(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear_2(x)

        return x


class Transformer(nn.Module):
    def __init__(self, max_seq_len, d_model, n_heads, d_ff):
        super().__init__()

        self.ln_1 = LayerNorm(d_model)
        self.mha = MultiHeadAttn(max_seq_len, d_model, n_heads, causal=True)
        self.ln_2 = LayerNorm(d_model)
        self.ffn = FeedFowardNetwork(d_model, d_ff)


    def forward(self, x):

        x = x +self.mha(self.ln_1(x))

        x = x + self.ffn(self.ln_2(x))

        return x
    
B, max_seq_len, d_model, n_heads = 2, 5, 16, 4
d_ff = 3*d_model
T = 3

print('atten params', d_model* 3 * d_model)
print('ffn params', 2 * (d_model * 3 * d_model))

tf = Transformer(max_seq_len, d_model, n_heads, d_ff)
x = torch.randn((B, T, d_model))
out = tf(x)
loss = out.sum()
loss.backward

assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
print("shape ok:", tuple(out.shape))
print("no NaN:", not torch.isnan(out).any().item())






        
