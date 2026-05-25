# Attempt 1 — May 23, 2026
# Started: 11:21am
# First pass: 12:57
# Correct: yes — shape preserved, causality property holds (perturbing x at pos t doesn't change out[:, :t, :])
# Stuck on: none
# Notes (bugs caught + fixes, with hints):
#   - causal_mask_additive was missing `self` — hidden until causal=True
#   - causal mask logic was inverted (col < row → masking past). flipped to col > row → mask future
#   - register_buffer had a typo ('casual_mask') AND was duplicating state with self.causal_mask.
#     fix: don't pre-assign — register_buffer creates the attribute. one line, no duplicate state.
#   - missing output projection nn.Linear(d_model, d_model) at the end of forward — heads couldn't mix
#   - cleaned Q/K/V split with qkv.chunk(3, dim=-1) instead of slicing in two different methods
#   - used reshape (not view) after permute since the tensor is non-contiguous after permute
#   - verification via shape assert + causality property (not allclose vs torch.nn.MHA, which won't match without copying weights and adding torch's out_proj convention)
import torch
import torch.nn as nn
import math

class MultiHeadAttn(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, causal=False):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        #print(self.seq_len)

        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        #print(self.d_head)

        self.qkv = nn.Linear(d_model, 3*d_model)

        if causal:
            causal_mask = self.causal_mask_additive(self.seq_len)
        else:
            causal_mask = torch.zeros((seq_len, seq_len))

        self.register_buffer('causal_mask', causal_mask)

        self.out_proj = nn.Linear(d_model, d_model)

    def causal_mask_additive(self, seq_len):
        row_indices = torch.arange(seq_len)[:, None]
        col_indices = torch.arange(seq_len)[None, :]
        mask = col_indices > row_indices
        #print(mask)
        causal_mask = torch.zeros((seq_len, seq_len)).masked_fill(mask, float('-inf'))
        #print(causal_mask)
        return causal_mask


    def causal_attn(self, q, k):
        #print(self.d_head)
        
        #print(q.shape)
        kt = k.transpose(3,2)
        #print(kt.shape)
        attn = (q @ kt) / math.sqrt(self.d_head)
        #print(attn.shape)
        #print(self.causal_mask)
        attn = attn + self.causal_mask
        causal_attn = torch.nn.functional.softmax(attn, dim=-1)
        return causal_attn
        

    def forward(self, x):
        B, T, d_model = x.shape
        #print(x.shape)
        qkv = self.qkv(x) # B, T, 3*d_model
        #print(qkv.shape)
        qkv = qkv.view(B, T, self.n_heads, 3*self.d_head) # B, T, H, D
        qkv = qkv.permute(0, 2, 1, 3)
        #print(qkv.shape)
        q, k, v = qkv.chunk(3, dim=-1)
        causal_attn = self.causal_attn(q, k)
        # print(causal_attn.shape)
        # print(v.shape)
        out = causal_attn @ v
        #print(out.shape)
        out = out.permute(0, 2, 1, 3)
        #print(out.shape)
        out = out.reshape(B, T, d_model)
        #print(out.shape)
        out = self.out_proj(out)
        return out

   

mha = MultiHeadAttn(5, 16, 4, True)
x = torch.randn((2, 5, 16))
out = mha(x)
# torch_out = torch.nn.MultiheadAttention(16, 4)(x)
# print(torch.allclose(out, torch_out))
x_new = x.clone()
x_new[:, 3, :] += 100
out_1 = mha(x_new) 
assert x.shape == out.shape
print(torch.allclose(out[:, :3, :], out_1[:, :3, :]))