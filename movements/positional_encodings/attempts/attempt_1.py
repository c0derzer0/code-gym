# Attempt 1 — sinusoidal: 2026-05-30 · RoPE: 2026-06-04
#
# Sinusoidal
#   Started: 6:31pm — First pass: 6:49pm (~18 min) — Correct after hints
#   Bugs caught (with hints):
#     - base was 1000 not 10000 (paper convention) — off by an order of magnitude
#     - the denominator was using `i` directly so adjacent sin/cos dims got
#       different frequencies. fix: use i_pair = i // 2 so each pair (2k, 2k+1)
#       shares the same θ_k.
#     - sin+phase trick is good: cos(x) = sin(x + π/2), so phase = (i % 2) * π/2
#       gives even=sin / odd=cos without a branch.
#   Verified: pe[0] = [0, 1, 0, 1, ...] exactly; pe[1] varies as expected.
#
# RoPE
#   Started: 10:35pm — Finished: 11:30pm (~55 min)
#   Verified via relative-position invariance: for q, k vectors held constant,
#   <RoPE(q,m), RoPE(k,n)> depends only on (m-n). Tested across 4 pairs with
#   diff=-3 → identical inner product (1.585057). That's the proof.
#
#   Concept locked: rotate pairs of dims (2k, 2k+1) by angle m·θ_k where
#   θ_k = 10000^(-2k/d). Apply to Q and K inside attention (after projection,
#   before dot product). V is NOT rotated. The dot product property
#   <Rotate(q,m), Rotate(k,n)> depends only on (m-n) gives genuinely relative
#   attention with an absolute-position implementation.
#
#   Clever vectorization (mine): instead of explicit even/odd split, build a
#   companion vector x' = [-x_1, x_0, -x_3, x_2, ...] via mask + stack-flatten,
#   then rope = x * cos + x' * sin. Mathematically equivalent to the explicit
#   per-pair rotation.
#
#   But — NOT efficient compared to the standard. My version:
#     - works on full-d tensors (2× FLOPs vs the half-d standard)
#     - uses (seq_len, d) cos/sin tables with duplicates (2× memory)
#     - more PyTorch eager kernel launches (mask mul, stack, two muls, add)
#   The standard half-d approach (and Llama's rotate-half variant) is ~2×
#   faster in eager mode and is what all production LLMs use.
#
#   Followups for attempt 2:
#     - Switch to Llama's rotate-half pattern: split x into halves, do
#       `x * cos + cat([-x[..., d/2:], x[..., :d/2]], dim=-1) * sin`.
#       Cleanest code, fastest, SIMD-friendly.
#     - Cache cos/sin tables as nn.Module buffers up to max_seq_len, slice
#       [:T, :] at forward time. Currently rebuilt every call.
#     - Wrap apply_rope in nn.Module so .to(device) moves the buffers.
#     - Test with V — confirm Q,K are rotated but V isn't.

import torch


def sinusoidal(seq_len, d_model):
    pos = torch.arange(seq_len)[:, None]
    i = torch.arange(d_model)[None, :]
    i_pair = i // 2
    sin_cos_phase = (i % 2) * torch.pi / 2  # 0 for even (sin), π/2 for odd (cos)
    return torch.sin(pos / 10000 ** (2 * i_pair / d_model) + sin_cos_phase)


def apply_rope(q, k, seq_len):
    """
    q, k: (..., seq_len, d_head)
    Returns rotated q, k of the same shape.
    """
    d_model = q.shape[-1]
    i = torch.arange(d_model)[None, :]
    i_pair = i // 2
    theta = 10000.0 ** -(2 * i_pair / d_model)
    pos = torch.arange(seq_len)[:, None]
    m_theta = pos * theta
    cos_m_theta = torch.cos(m_theta)
    sin_m_theta = torch.sin(m_theta)

    def companion(mat):
        # Build x' = [-x_1, x_0, -x_3, x_2, ...] so that x * cos + x' * sin
        # implements the per-pair 2D rotation in one go.
        mask = torch.ones(d_model)[None, :] + (i % 2 == 1) * -2  # [1, -1, 1, -1, ...]
        mat_flipped = mat * mask
        evens = mat_flipped[..., 0::2]
        odds = mat_flipped[..., 1::2]
        return torch.stack((odds, evens), dim=-1).flatten(start_dim=-2)

    q_rope = q * cos_m_theta + companion(q) * sin_m_theta
    k_rope = k * cos_m_theta + companion(k) * sin_m_theta
    return q_rope, k_rope


if __name__ == "__main__":
    # Sinusoidal sanity: pe[0] should be [0, 1, 0, 1, ...] exactly.
    pe = sinusoidal(4, 8)
    assert pe.shape == (4, 8)
    assert torch.allclose(pe[0], torch.tensor([0.0, 1.0] * 4))
    print("sinusoidal pe[0]:", pe[0].tolist())
    print("sinusoidal pe[1]:", [round(v, 4) for v in pe[1].tolist()])

    # RoPE proof: <RoPE(q,m), RoPE(k,n)> depends only on (m-n).
    torch.manual_seed(0)
    d_head = 8
    q_vec = torch.randn(d_head)
    k_vec = torch.randn(d_head)

    def inner_at(m, n):
        seq_len = max(m, n) + 1
        q = q_vec[None, :].repeat(seq_len, 1)
        k = k_vec[None, :].repeat(seq_len, 1)
        qr, kr = apply_rope(q, k, seq_len)
        return (qr[m] * kr[n]).sum().item()

    print("\nRoPE relative-position test (all have m-n = -3):")
    inners = []
    for m, n in [(2, 5), (4, 7), (10, 13), (20, 23)]:
        val = inner_at(m, n)
        inners.append(val)
        print(f"  m={m:2d}, n={n:2d}, diff={m-n}, inner={val:.6f}")
    assert max(inners) - min(inners) < 1e-5
    print("  ✓ all identical within 1e-5 — relative-position invariance holds")
