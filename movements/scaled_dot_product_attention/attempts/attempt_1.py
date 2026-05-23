# Attempt 1 — May 22, 2026
# Started: 7:17pm
# finished: pass 1 - 8:10
# First pass (runs without error): no
# Correct (passes tests): 
# Stuck on: mask for causal attention
# Notes: have the correct idea, but did not get the correct output for the mask

import torch

def attention(q, k, v, use_mask=False):
    # q, k, v - shape [B, T, D]
    # 
    B, T, D = q.shape
    kt = k.transpose(1, 2)
    print(kt.shape)
    attn = (q @ kt) / torch.sqrt(torch.tensor(D))
    print(attn.shape) # attn = B, T, T
    mask = torch.ones_like(attn)
    if use_mask:
        lower = torch.tril(mask, diagonal=1)
        print(lower)
        upper = torch.triu(mask * float('-inf'), diagonal=1)
        print(upper)
        mask = lower + upper
    print(mask)
    causal_attn = torch.nn.functional.softmax(attn * mask, dim=-1)
    print(causal_attn)




    
q = torch.randn(2, 8, 64)
k = torch.randn(2, 8, 64)
v = torch.randn(2, 8, 64)
attention(q, k, v, True)


