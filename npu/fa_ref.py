"""FlashAttention-2 forward-pass reference in pure Python.

This mirrors the block-at-a-time algorithm our AIE kernel will execute, so the
kernel has a bit-for-bit (modulo bf16 rounding) correctness target. Written in
plain torch, no tricks. Includes a test comparing against the standard-attention
output torch produces inside smollm.py.

Notation (FA-2 paper, Alg. 1):

    Q : [B, H, T_q, Dh]       queries
    K : [B, H, T_k, Dh]       keys
    V : [B, H, T_k, Dh]       values
    O : [B, H, T_q, Dh]       output

The algorithm tiles Q into blocks of Br rows and K/V into blocks of Bc rows:

    For each query block Q_i  (rows [i*Br, (i+1)*Br)):
      m_i = -inf(Br)                       # running row-max
      l_i = 0(Br)                          # running row-sum-of-exp
      O_i = 0(Br, Dh)                      # running output accumulator
      For each kv block (K_j, V_j):
        S_ij = (Q_i @ K_j^T) / sqrt(Dh)    # [Br, Bc]
        if causal: mask out j>i positions  # -inf
        m_new = max(m_i, rowmax(S_ij))     # [Br]
        alpha = exp(m_i - m_new)           # [Br]  (rescale factor)
        P_ij  = exp(S_ij - m_new)          # [Br, Bc]
        l_i   = alpha * l_i + rowsum(P_ij)
        O_i   = diag(alpha) @ O_i + P_ij @ V_j
        m_i   = m_new
      O_i = O_i / l_i                      # final normalize
      store O_i

The two matmuls (S = QK^T and O += PV) are the heavy compute; the softmax
stats (m, l, rescale) are O(Br) per kv block. No T^2 intermediate materialises
in memory — we only ever hold one [Br, Bc] S_ij / P_ij tile at a time.
"""
from __future__ import annotations
import math
import torch


def flash_attention_ref(
    Q: torch.Tensor,      # [B, H, Tq, D]
    K: torch.Tensor,      # [B, H, Tk, D]
    V: torch.Tensor,      # [B, H, Tk, D]
    Br: int,
    Bc: int,
    causal: bool = True,
    start_pos: int = 0,   # absolute position of Q[0] in the full sequence
    accum_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Block-at-a-time FA-2 forward, mirroring what the AIE kernel will do.

    Runs entirely on CPU, fp32 internally (matches AIE accumulators). Inputs
    can be bf16; Q/K/V are upcast for matmul, output is returned in accum_dtype.

    The causal mask uses absolute positions:
      row r in Q corresponds to seq pos (start_pos + r)
      col c in K corresponds to seq pos c
      masked iff c > start_pos + r
    """
    B, H, Tq, D = Q.shape
    _, _, Tk, _ = K.shape
    assert K.shape == (B, H, Tk, D) and V.shape == (B, H, Tk, D)
    # Tail blocks are handled by clipping r1/c1 to Tq/Tk in the loop, so Tq and
    # Tk are NOT required to be multiples of Br/Bc.

    inv_sqrt_d = 1.0 / math.sqrt(D)
    # Promote inputs to the accumulator dtype; the AIE kernel does bf16 in / f32
    # accumulate inside each matmul tile, which this matches.
    Qf = Q.to(accum_dtype)
    Kf = K.to(accum_dtype)
    Vf = V.to(accum_dtype)

    O = torch.zeros((B, H, Tq, D), dtype=accum_dtype, device=Q.device)

    # Work out the block grid. Pad-free version: handle tail blocks with the
    # actual remainder count (the AIE kernel will need to handle this too).
    n_qblocks = (Tq + Br - 1) // Br
    n_kblocks = (Tk + Bc - 1) // Bc

    for b in range(B):
        for h in range(H):
            for i in range(n_qblocks):
                r0, r1 = i * Br, min((i + 1) * Br, Tq)
                br = r1 - r0
                Qi = Qf[b, h, r0:r1]                        # [br, D]

                m = torch.full((br,), float("-inf"), dtype=accum_dtype, device=Q.device)
                l = torch.zeros((br,), dtype=accum_dtype, device=Q.device)
                Oi = torch.zeros((br, D), dtype=accum_dtype, device=Q.device)

                for j in range(n_kblocks):
                    c0, c1 = j * Bc, min((j + 1) * Bc, Tk)
                    bc = c1 - c0
                    Kj = Kf[b, h, c0:c1]                    # [bc, D]
                    Vj = Vf[b, h, c0:c1]                    # [bc, D]

                    # S_ij = Q_i @ K_j^T * (1/sqrt(D))
                    S = (Qi @ Kj.transpose(0, 1)) * inv_sqrt_d    # [br, bc]

                    if causal:
                        # rows r in [r0, r1) correspond to absolute pos (start_pos + r)
                        # cols c in [c0, c1) correspond to absolute pos c
                        row_pos = torch.arange(r0, r1, device=Q.device) + start_pos  # [br]
                        col_pos = torch.arange(c0, c1, device=Q.device)              # [bc]
                        mask = col_pos[None, :] > row_pos[:, None]                   # [br, bc]
                        S = S.masked_fill(mask, float("-inf"))

                    # Online softmax update
                    m_new = torch.maximum(m, S.max(dim=-1).values)    # [br]
                    # If a row has all-masked S (can happen for a Q row with no
                    # keys attended to yet, in left-padded decode), m_new is
                    # -inf and exp(S - m_new) = 0; guard against NaN later.
                    alpha = torch.exp(m - m_new)                      # [br]
                    # exp(-inf - (-inf)) = exp(nan) = nan; 0 out those rows
                    alpha = torch.where(torch.isneginf(m_new), torch.zeros_like(alpha), alpha)
                    P = torch.exp(S - m_new[:, None])                 # [br, bc]
                    P = torch.where(torch.isneginf(m_new)[:, None], torch.zeros_like(P), P)

                    l = alpha * l + P.sum(dim=-1)                     # [br]
                    Oi = alpha[:, None] * Oi + P @ Vj                 # [br, D]
                    m = m_new

                # Final normalise; rows with l==0 (fully masked) stay 0.
                safe_l = torch.where(l > 0, l, torch.ones_like(l))
                Oi = Oi / safe_l[:, None]
                Oi = torch.where(l[:, None] > 0, Oi, torch.zeros_like(Oi))
                O[b, h, r0:r1] = Oi

    return O


def _torch_causal_attention(Q, K, V, start_pos: int = 0):
    """Standard-attention reference (what smollm.py computes on CPU path)."""
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    S = torch.matmul(Q.float(), K.float().transpose(-2, -1)) / math.sqrt(D)
    # Causal mask: row r in Q (abs pos = start_pos+r) attends to cols [0, start_pos+r]
    row = torch.arange(Tq, device=Q.device)[:, None] + start_pos
    col = torch.arange(Tk, device=Q.device)[None, :]
    mask = torch.where(col <= row, 0.0, float("-inf")).to(S.dtype)
    S = S + mask
    A = torch.softmax(S, dim=-1)
    return torch.matmul(A, V.float())


def _self_test():
    torch.manual_seed(0)
    B, H = 1, 9
    for Tq, Tk, D in [(16, 16, 64), (32, 32, 64), (17, 17, 64), (64, 64, 64), (5, 20, 64)]:
        Q = torch.randn(B, H, Tq, D, dtype=torch.bfloat16)
        K = torch.randn(B, H, Tk, D, dtype=torch.bfloat16)
        V = torch.randn(B, H, Tk, D, dtype=torch.bfloat16)

        # Figure out start_pos consistent with Tq != Tk. When Tq < Tk we're in a
        # "prefix is already cached, new chunk extends it" scenario.
        start_pos = Tk - Tq

        Oref = _torch_causal_attention(Q, K, V, start_pos=start_pos)
        for Br, Bc in [(8, 16), (16, 32), (32, 32)]:
            Ofa = flash_attention_ref(Q, K, V, Br=Br, Bc=Bc, causal=True, start_pos=start_pos)
            diff = (Ofa - Oref).abs()
            print(f"  Tq={Tq:3d} Tk={Tk:3d} D={D}  Br={Br:2d} Bc={Bc:2d}  "
                  f"max|Δ|={diff.max().item():.3e}  mean|Δ|={diff.mean().item():.3e}")
            assert diff.max() < 1e-5, "FA ref must match standard attention in fp32"
    print("FA reference matches standard-attention to fp32 rounding.")


if __name__ == "__main__":
    _self_test()
