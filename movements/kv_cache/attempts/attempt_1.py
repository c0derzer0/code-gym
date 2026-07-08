# Attempt 1 — Jul 6, 2026 (finished 2026-07-07)
# Started: 6:07pm
# First pass (MHA structure ran): 7:20pm
# Correct (cached generation matches full-sequence to 1e-5): 7:45pm (next-day)
# Stuck on: causal mask that produced same output only in cached case, not prefill
# Notes (bugs caught + fixes, with hints):
#   - missing super().__init__() in nn.Module subclass — Linear wouldn't register
#   - x @ self.qkv (tried to matmul a tensor with a Module). fix: self.qkv(x)
#   - .permute() called with runtime shape values (B, T, n_heads, d_head) instead
#     of axis indices (0, 2, 1, 3). same bug appeared 4 times before fix, then
#     reappeared once on the final permute after attention output.
#   - torch.cat with cache['k'] when cache=None → None subscript error. fix:
#     branch on cache before concat, and init cache={} in the None case before
#     writing cache['k'] = k / cache['v'] = v.
#   - torch.cat along dim=0 (batch dim) instead of dim=-2 (seq/time dim).
#     grew batch instead of extending along sequence.
#   - k.T on a 4D tensor. .T transposes ALL dims, so (B,H,T,d_head) → (d_head,T,H,B).
#     fix: k.permute(0, 1, 3, 2) or k.transpose(-2, -1).
#   - scaled by math.sqrt(d_model) instead of math.sqrt(d_head). MHA scale factor
#     is per-head, not model-wide.
#   - causal mask returned as bool tensor. bool.masked_fill(bool, float('-inf'))
#     stays bool (bool(-inf)=True), so mask ended up 0/1 instead of 0/-inf.
#     attn + bool broadcast added 1.0 to future positions instead of masking them.
#     the bug hid in the prefill path (T=full) but not the cached path (T_new=1)
#     because the T_new=1 mask row is all-allowed anyway. surfaced only when
#     comparing prefill vs cached outputs. fix: start from zeros tensor, then
#     masked_fill with the bool condition and float('-inf'). OR use torch.where.
#   - nn.linear (lowercase) vs nn.Linear typo. runtime AttributeError.
#   - forgot output projection at first. added self.out_proj = nn.Linear(d_model, d_model)
#     and applied after the attention output reshape.
#   - if cache: works but is truthiness-based; switched to if cache is not None
#     as followup (avoids empty-dict edge case).
# Verified: full-sequence forward matches token-by-token cached generation to 1e-5,
# and cache['k'].shape = (B, H, T, d_head) at the end.
# Followups for attempt 2:
#   - don't mutate the input cache dict; build new_cache locally and return it.
#   - track T_past by explicit shape read (cleaner than the current if/else).
#   - benchmark: measure prefill throughput vs one-token-at-a-time cached generation
#     time. Show the O(T^2) → O(T) speedup mechanically with wall-time numbers.

from torch import nn
import torch
import math

class MultiHeadAttnWithKVCache(nn.Module):
    def __init__(self, max_seq_len, d_model, n_heads, causal=True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def causal_mask(self, T, T_past):
        col_ind = torch.arange(self.max_seq_len)[None, :]
        row_ind = torch.arange(self.max_seq_len)[:, None]
        mask = (col_ind > row_ind) 
        causal_mask = torch.zeros((self.max_seq_len, self.max_seq_len))
        causal_mask = causal_mask.masked_fill(mask, float('-inf'))
        print(causal_mask)

        return causal_mask[T_past:T_past+T, :T_past+T]

    def compute_attn(self, q, k, T, T_past):  # B, T, d_model, B, T, d_model

        attn = q @ k.permute(0, 1, 3, 2) # B, T, T
        attn = attn / math.sqrt(self.d_head)
        mask = self.causal_mask(T, T_past)
        attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        
        return attn # B, n-head, T, T
    

    def forward(self, x, cache=None):
        """
        x:     (B, T_new, d_model) — during generation, T_new == 1
        cache: {'k': (B, H, T_past, d_head), 'v': (B, H, T_past, d_head)}
               or None (first step)

        Returns: (out, new_cache)
          out:       (B, T_new, d_model)
          new_cache: {'k': (B, H, T_past + T_new, d_head), 'v': same shape}
        """

        B, T, d_model = x.shape
        

        qkv = self.qkv(x)

        q, k, v = torch.split(qkv, d_model, dim=-1)

        q = q.reshape(B, T, self.n_heads, self.d_head)
        q = q.permute(0, 2, 1, 3)

        k = k.reshape(B, T, self.n_heads, self.d_head)
        k = k.permute(0, 2, 1, 3)

        v = v.reshape(B, T, self.n_heads, self.d_head)
        v = v.permute(0, 2, 1, 3) # B, n_head, T, d_head

        if cache:
            _, _, T_past, _ = cache['k'].shape
            k = torch.cat((cache['k'], k), dim=-2)
            v = torch.cat((cache['v'], v), dim=-2)
        else:
            T_past = 0
            cache = {}


        cache['k'] = k
        cache['v'] = v

        attn = self.compute_attn(q, k, T, T_past) # B, n-head, T, T

        out = attn @ v

        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B, T, d_model)
        out = self.out_proj(out)

        return out, cache



if __name__ == "__main__":
      torch.manual_seed(0)
      B, T, d_model, n_heads = 1, 3, 4, 2
      mha = MultiHeadAttnWithKVCache(max_seq_len=5, d_model=d_model, n_heads=n_heads)

      x = torch.randn(B, T, d_model)

      with torch.no_grad():
          out_full, _ = mha(x, cache=None)

      cache = None
      outs = []
      with torch.no_grad():
          for t in range(T):
              out_t, cache = mha(x[:, t:t+1, :], cache=cache)
              outs.append(out_t)
      out_cached = torch.cat(outs, dim=1)

      print("shapes match:", out_full.shape == out_cached.shape)
      print("values match:", torch.allclose(out_full, out_cached, atol=1e-5))
      print("cache shape:", cache['k'].shape)
