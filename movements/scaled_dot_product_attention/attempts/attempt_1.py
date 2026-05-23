# Attempt 1 — May 22, 2026
# Started: 7:17pm
# finished: pass 1 - 8:10
# finished: pass 2 - 9:00 with hints
# First pass (runs without error): no
# second pass (runs without error): yes
# Correct (passes tests): 
# Stuck on: mask for causal attention - figured on second pass
# Notes: have the correct idea, but did not get the correct output for the mask. everyhting works well in second pass after hints

import torch
import math

def attention(q, k, v, use_mask=False):
    # q, k, v - shape [B, T, D]
    # 
    B, T, D = q.shape
    kt = k.transpose(1, 2)
    #print(kt.shape)
    attn = (q @ kt) / math.sqrt(D)
    #print(attn.shape) # attn = B, T, T
    mask = torch.ones_like(attn)
    if use_mask:
        mask = torch.triu(mask, diagonal=1)
        casual_mask = mask == 1
        mask = mask.masked_fill(casual_mask, float('-inf'))
    causal_attn = torch.nn.functional.softmax(attn + mask, dim=-1)
    return causal_attn @ v

    
def attention_mask_optimized(q, k, v, use_mask=False):
    # q, k, v - shape [B, T, D]
    # 
    B, T, D = q.shape
    kt = k.transpose(1, 2)
    attn = (q @ kt) /  math.sqrt(D)
    if use_mask:
        row_indices = torch.arange(attn.size(1)).unsqueeze(1)
        col_indices = torch.arange(attn.size(2)).unsqueeze(0)
        attn = attn.masked_fill(col_indices>row_indices, float('-inf'))
    causal_attn = torch.nn.functional.softmax(attn, dim=-1)
    return causal_attn @ v




    
q = torch.randn(2, 8, 64)
k = torch.randn(2, 8, 64)
v = torch.randn(2, 8, 64)
out_1 = attention(q, k, v, True)
out_2 = attention_mask_optimized(q, k, v, True)
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

print(torch.allclose(out_1, ref, atol=1e-5))
print(torch.allclose(out_2, ref, atol=1e-5))
