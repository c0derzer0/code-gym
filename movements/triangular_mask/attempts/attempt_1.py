# Attempt 1 — May 23, 2026
# Started: 11am
# finished: 11:11
# First pass (runs without error): yes
# Correct (passes tests): yes
# Stuck on:
# Notes:
import torch

def causal_mask(seq_len: int) -> torch.Tensor:
    row_indices = torch.arange(seq_len)[:, None]
    col_indices = torch.arange(seq_len)[None, :]
    mask = col_indices <= row_indices
    print(mask)
    return mask

def causal_mask_additive(seq_len: int) -> torch.Tensor:
    mask = causal_mask(seq_len)
    mask_addtive = torch.zeros((seq_len, seq_len))
    mask_addtive = mask_addtive.masked_fill(~mask, float('-inf'))
    print(mask_addtive)
    return mask_addtive

causal_mask(4)
causal_mask_additive(4)