"""Qwen3.5/3.6 MoE forward pass — CPU reference.

Architecture (per inspect_gguf.py + gguf's qwen35moe tensor map):
- 40 blocks = 10 gated-attention @ [3,7,11,...,39] + 30 Gated-DeltaNet SSM.
- Gated attention: attn_q projects to 2*n_head*head_dim = [q | q_gate],
  attn output is q_gate-modulated and then projected by attn_output
  (in_dim = n_head*head_dim). n_head=16, head_dim=256, n_kv=2, GQA 8x.
  Per-head Q/K RMSNorm. Partial RoPE on first rope_dim=64 of 256.
- Gated DeltaNet SSM: attn_qkv projects to [q | k | v], depthwise conv1d
  (k=4) across the 8192 mixed channels, per-head α/β/dt from input, recurrent
  state update, gated by attn_gate, projected by ssm_out. Not yet implemented.
- Every block has a MoE block (256 experts, top-8) + a shared expert.
  Router: ffn_gate_inp (2048→256) gates across experts; weighted top-8.
  Each expert is a SwiGLU: down(silu(gate) * up).
  Shared expert is always active, gated by sigmoid(ffn_gate_inp_shexp · x).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from qwen.model import Config, TensorStore, from_bf16


# ---------------------------------------------------------------------------
# Weight loading: GGML-order numpy → torch. GGUF stores weights in
# (in, out) order; torch's F.linear expects (out, in). We transpose on load.
# ---------------------------------------------------------------------------

def _w(ts: TensorStore, name: str, dtype=torch.float32) -> torch.Tensor:
    arr_fp32 = ts.get(name, dtype="fp32", keep=False)
    # ggml (in, out) → torch (out, in); contiguous copy so torch owns writable mem.
    t = torch.from_numpy(np.ascontiguousarray(arr_fp32.T).copy())
    return t.to(dtype)


def _v(ts: TensorStore, name: str, dtype=torch.float32) -> torch.Tensor:
    arr_fp32 = ts.get(name, dtype="fp32", keep=False)
    return torch.from_numpy(arr_fp32.copy()).to(dtype)


# ---------------------------------------------------------------------------
# Norms and RoPE
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    rms = x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return ((x_f * rms) * w.float()).to(x.dtype)


def build_partial_rope_cache(
    rope_dim: int, max_pos: int, theta: float,
    dtype=torch.float32, device="cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables of shape [max_pos, rope_dim]. Only the first rope_dim
    lanes of each head_dim participate in rotation; the remaining head_dim -
    rope_dim lanes pass through unchanged."""
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
        / rope_dim
    ))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int,
) -> torch.Tensor:
    """x: [B, T, H, head_dim]. cos/sin: [T, rope_dim]. Rotate x[..., :rope_dim]."""
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    d2 = rope_dim // 2
    x1, x2 = x_rot[..., :d2], x_rot[..., d2:]
    rot = torch.cat([-x2, x1], dim=-1)
    # Broadcast cos/sin across B (dim 0) and H (dim 2): [1, T, 1, rope_dim].
    cos_b = cos.unsqueeze(0).unsqueeze(2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)
    x_rot_out = x_rot * cos_b + rot * sin_b
    return torch.cat([x_rot_out, x_pass], dim=-1)


# ---------------------------------------------------------------------------
# Attention layer (gated)
# ---------------------------------------------------------------------------

@dataclass
class AttnLayer:
    i: int
    attn_norm: torch.Tensor             # [D]
    post_norm: torch.Tensor             # [D]
    wq: torch.Tensor                    # [2*n_head*head_dim, D]
    wk: torch.Tensor                    # [n_kv*head_dim, D]
    wv: torch.Tensor                    # [n_kv*head_dim, D]
    wo: torch.Tensor                    # [D, n_head*head_dim]
    q_norm: torch.Tensor                # [head_dim]
    k_norm: torch.Tensor                # [head_dim]

    @classmethod
    def load(cls, ts: TensorStore, i: int) -> "AttnLayer":
        p = f"blk.{i}."
        return cls(
            i=i,
            attn_norm=_v(ts, p + "attn_norm.weight"),
            post_norm=_v(ts, p + "post_attention_norm.weight"),
            wq=_w(ts, p + "attn_q.weight"),
            wk=_w(ts, p + "attn_k.weight"),
            wv=_w(ts, p + "attn_v.weight"),
            wo=_w(ts, p + "attn_output.weight"),
            q_norm=_v(ts, p + "attn_q_norm.weight"),
            k_norm=_v(ts, p + "attn_k_norm.weight"),
        )


def attn_forward(
    x: torch.Tensor, layer: AttnLayer, cfg: Config,
    cos: torch.Tensor, sin: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
    start_pos: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Gated GQA with partial RoPE and per-head Q/K RMSNorm.

    Returns (output [B,T,D], new kv cache).
    """
    B, T, D = x.shape
    Hq, Hkv, Dh, rd = cfg.n_head, cfg.n_head_kv, cfg.head_dim, cfg.rope_dim

    h = rms_norm(x, layer.attn_norm, cfg.rms_eps)

    # Gated Q with *interleaved per-head* layout:
    #   [q_h0 | gate_h0 | q_h1 | gate_h1 | ... | q_h{Hq-1} | gate_h{Hq-1}]
    # (llama.cpp qwen35moe.cpp uses stride n_embd_head*2 per head; gate view
    # is offset by n_embd_head within each slot.)
    q_full = F.linear(h, layer.wq)           # [B, T, 2*Hq*Dh]
    q_full = q_full.view(B, T, Hq, 2, Dh)
    q = q_full[..., 0, :]                     # [B, T, Hq, Dh]
    q_gate = q_full[..., 1, :]                # [B, T, Hq, Dh]
    k = F.linear(h, layer.wk).view(B, T, Hkv, Dh)
    v = F.linear(h, layer.wv).view(B, T, Hkv, Dh)

    # Per-head Q/K RMSNorm (over head_dim).
    q = rms_norm(q, layer.q_norm, cfg.rms_eps)
    k = rms_norm(k, layer.k_norm, cfg.rms_eps)

    # Partial RoPE at absolute positions start_pos..start_pos+T.
    cos_slice = cos[start_pos: start_pos + T]  # [T, rope_dim]
    sin_slice = sin[start_pos: start_pos + T]
    q = apply_partial_rope(q, cos_slice, sin_slice, rd)
    k = apply_partial_rope(k, cos_slice, sin_slice, rd)

    # GQA repeat: [B, T, Hkv, Dh] -> [B, T, Hq, Dh] by repeating each KV head.
    rep = Hq // Hkv
    k = k.repeat_interleave(rep, dim=2)
    v = v.repeat_interleave(rep, dim=2)

    # KV cache concat (B, T_cached + T, Hq, Dh).
    if kv_cache is not None:
        k_prev, v_prev = kv_cache
        k = torch.cat([k_prev, k], dim=1)
        v = torch.cat([v_prev, v], dim=1)
    new_cache = (k, v)

    # Scaled dot-product attention with causal mask. [B, Hq, Tq, Tk].
    q_btHD = q.transpose(1, 2)                  # [B, Hq, T, Dh]
    k_btHD = k.transpose(1, 2)                  # [B, Hq, Tk, Dh]
    v_btHD = v.transpose(1, 2)
    scores = torch.matmul(q_btHD, k_btHD.transpose(-2, -1)) / (Dh ** 0.5)

    Tk = k_btHD.shape[-2]
    # Causal mask: row t (absolute start_pos + t) attends to cols <= start_pos + t.
    if T > 1:
        rows = torch.arange(T) + start_pos
        cols = torch.arange(Tk)
        mask = cols[None, :] > rows[:, None]       # [T, Tk]
        scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn, v_btHD)             # [B, Hq, T, Dh]
    out = out.transpose(1, 2).contiguous()        # [B, T, Hq, Dh]

    # Gated attention: multiply by sigmoid(q_gate) per-head per-dim before wo.
    # (qwen35moe.cpp: gate_sigmoid = ggml_sigmoid(gate); cur = cur * gate_sigmoid)
    out = out * torch.sigmoid(q_gate)             # [B, T, Hq, Dh]
    out = out.view(B, T, Hq * Dh)
    out = F.linear(out, layer.wo)

    return out, new_cache


# ---------------------------------------------------------------------------
# SSM layer (Gated DeltaNet)
# ---------------------------------------------------------------------------

@dataclass
class SSMLayer:
    i: int
    attn_norm: torch.Tensor
    post_norm: torch.Tensor
    w_in: torch.Tensor                   # attn_qkv: [8192, D]
    w_gate: torch.Tensor                 # attn_gate: [4096, D]
    conv1d: torch.Tensor                 # ssm_conv1d: [8192, 4] — per-channel kernel
    a: torch.Tensor                      # ssm_a: [n_v_heads=32]
    alpha: torch.Tensor                  # ssm_alpha: [n_v_heads, D]
    beta: torch.Tensor                   # ssm_beta: [n_v_heads, D]
    dt_bias: torch.Tensor                # ssm_dt.bias: [n_v_heads]
    ssm_norm: torch.Tensor               # [head_v_dim=128]
    w_out: torch.Tensor                  # ssm_out: [D, d_inner=4096]

    @classmethod
    def load(cls, ts: TensorStore, i: int) -> "SSMLayer":
        p = f"blk.{i}."
        # ssm_conv1d is (4, 8192) in GGML order → transpose to (8192, 4) so
        # the kernel dim is last, matching torch F.conv1d's grouped layout.
        conv_raw = _v(ts, p + "ssm_conv1d.weight")  # (4, 8192)
        conv = conv_raw.T.contiguous()              # (8192, 4)
        return cls(
            i=i,
            attn_norm=_v(ts, p + "attn_norm.weight"),
            post_norm=_v(ts, p + "post_attention_norm.weight"),
            w_in=_w(ts, p + "attn_qkv.weight"),
            w_gate=_w(ts, p + "attn_gate.weight"),
            conv1d=conv,
            a=_v(ts, p + "ssm_a"),
            alpha=_w(ts, p + "ssm_alpha.weight"),
            beta=_w(ts, p + "ssm_beta.weight"),
            dt_bias=_v(ts, p + "ssm_dt.bias"),
            ssm_norm=_v(ts, p + "ssm_norm.weight"),
            w_out=_w(ts, p + "ssm_out.weight"),
        )


def _l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-vector L2 normalization along the last dim (matches ggml_l2_norm).
    Note this is *not* RMSNorm — no learnable scale, and it normalizes by
    sqrt(sum(x^2)), not sqrt(mean(x^2))."""
    return x / (x.pow(2).sum(-1, keepdim=True).clamp_min(eps).sqrt())


def ssm_forward(
    x: torch.Tensor, layer: SSMLayer, cfg: Config,
    ssm_state: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Qwen3-Next Gated DeltaNet.

    Implements the autoregressive (per-token) recurrence from
    delta-net-base.cpp:build_delta_net_autoregressive, looped over T for
    the prefill path. This is correct but slow for long prefills; the
    proper chunked prefill (build_delta_net_chunking) is a later
    optimization.

    State is (conv_state, recurrent_state):
      - conv_state: [B, n_ch=8192, conv_kernel-1=3]   — last 3 pre-conv inputs
      - recurrent_state: [B, H=32, K=128, V=128]      — per-head K×V matrix
    """
    B, T, D = x.shape
    Hv = cfg.ssm_n_v_heads       # 32
    Hk = cfg.ssm_n_k_heads       # 16
    Sk = cfg.ssm_d_state         # 128  (head_k_dim == head_v_dim)
    Sv = cfg.ssm_d_state
    Ck = Hk * Sk                 # 2048 — per-projection K dim
    d_inner = cfg.ssm_d_inner    # 4096
    conv_ch = 2 * Ck + d_inner   # 8192
    conv_k = cfg.ssm_d_conv      # 4
    rep = Hv // Hk               # 2 — K heads broadcast to V heads

    # --- Input projections ---------------------------------------------------
    qkv_mixed = F.linear(x, layer.w_in)          # [B, T, 8192]
    z = F.linear(x, layer.w_gate)                # [B, T, 4096]

    # --- α, β per head per token --------------------------------------------
    # alpha_proj → (B, T, Hv); add dt_bias; softplus; multiply by a_coef.
    alpha_raw = F.linear(x, layer.alpha)          # [B, T, Hv]
    beta_raw = F.linear(x, layer.beta)            # [B, T, Hv]
    alpha_biased = alpha_raw + layer.dt_bias
    alpha_sp = F.softplus(alpha_biased)
    g = alpha_sp * layer.a                        # [B, T, Hv]   — the "log decay"
    beta = torch.sigmoid(beta_raw)                # [B, T, Hv]

    # --- Depthwise conv1d with state -----------------------------------------
    # qkv_mixed along T convolved with layer.conv1d (per-channel k=4).
    # conv1d shape: (conv_ch, conv_k). We need torch.nn.functional.conv1d
    # with groups=conv_ch, weight shape (conv_ch, 1, conv_k).
    if ssm_state is None:
        conv_state = torch.zeros(B, conv_ch, conv_k - 1, dtype=x.dtype)
        rec_state = torch.zeros(B, Hv, Sk, Sv, dtype=x.dtype)
    else:
        conv_state, rec_state = ssm_state

    # Transpose qkv_mixed to [B, conv_ch, T] for conv.
    qkv_bt = qkv_mixed.transpose(1, 2).contiguous()        # [B, conv_ch, T]
    conv_in = torch.cat([conv_state, qkv_bt], dim=-1)      # [B, conv_ch, T + k-1]
    w_conv = layer.conv1d.unsqueeze(1)                      # [conv_ch, 1, conv_k]
    conv_out = F.conv1d(conv_in, w_conv, groups=conv_ch)    # [B, conv_ch, T]
    # Update conv_state: last (conv_k - 1) columns of conv_in.
    new_conv_state = conv_in[..., -(conv_k - 1):].contiguous()
    conv_out = F.silu(conv_out)

    # Back to [B, T, conv_ch].
    conv_out = conv_out.transpose(1, 2).contiguous()        # [B, T, 8192]

    # --- Split into q, k, v --------------------------------------------------
    q = conv_out[..., :Ck].view(B, T, Hk, Sk)               # [B, T, Hk, Sk]
    k = conv_out[..., Ck:2 * Ck].view(B, T, Hk, Sk)
    v = conv_out[..., 2 * Ck:].view(B, T, Hv, Sv)           # [B, T, Hv, Sv]

    # L2-normalize q, k per-head.
    q = _l2_norm(q, cfg.rms_eps)
    k = _l2_norm(k, cfg.rms_eps)

    # Broadcast K heads to V heads (Hk -> Hv, repeat factor `rep`).
    q = q.repeat_interleave(rep, dim=2)                     # [B, T, Hv, Sk]
    k = k.repeat_interleave(rep, dim=2)                     # [B, T, Hv, Sk]

    # --- DeltaNet recurrence (autoregressive loop) ---------------------------
    # scale q by 1/sqrt(Sk) up-front (build_delta_net scales q).
    scale = 1.0 / (Sk ** 0.5)
    q = q * scale

    S = rec_state                                            # [B, Hv, Sk, Sv]
    outs = []
    for t in range(T):
        qt = q[:, t]                                         # [B, Hv, Sk]
        kt = k[:, t]
        vt = v[:, t]                                         # [B, Hv, Sv]
        gt = g[:, t]                                         # [B, Hv]
        bt = beta[:, t]                                      # [B, Hv]

        # Decay state.
        S = S * torch.exp(gt)[..., None, None]

        # Delta rule: d = (v - S^T·k) * β; S += k ⊗ d.
        #   S: [B, Hv, Sk, Sv],  k: [B, Hv, Sk] → S^T·k is [B, Hv, Sv].
        sk = torch.einsum("bhkv,bhk->bhv", S, kt)
        d = (vt - sk) * bt[..., None]                        # [B, Hv, Sv]
        S = S + torch.einsum("bhk,bhv->bhkv", kt, d)

        # Output o = S^T · q.
        ot = torch.einsum("bhkv,bhk->bhv", S, qt)            # [B, Hv, Sv]
        outs.append(ot)

    o = torch.stack(outs, dim=1)                             # [B, T, Hv, Sv]

    # --- Gated norm: rmsnorm(o, ssm_norm) * silu(z) -------------------------
    # ssm_norm applies over the last dim (Sv=128) per head.
    o = rms_norm(o, layer.ssm_norm, cfg.rms_eps)             # [B, T, Hv, Sv]
    o = o.reshape(B, T, Hv * Sv)                             # [B, T, 4096]
    o = o * F.silu(z)                                        # [B, T, 4096]

    # --- Output projection ---------------------------------------------------
    out = F.linear(o, layer.w_out)                           # [B, T, D]

    return out, (new_conv_state, S)


# ---------------------------------------------------------------------------
# MoE layer
# ---------------------------------------------------------------------------

@dataclass
class MoELayer:
    i: int
    # Router
    w_router: torch.Tensor               # [n_expert, D]
    # Shared expert (always active, 1 SwiGLU)
    w_gate_sh: torch.Tensor              # [ff, D]
    w_up_sh: torch.Tensor                # [ff, D]
    w_down_sh: torch.Tensor              # [D, ff]
    w_gate_inp_sh: torch.Tensor          # [D] — scalar gate on shared expert
    # Per-expert stacked weights, kept in fp32 (large — TODO: lazy per-expert)
    w_gate_exps: torch.Tensor            # [n_expert, ff, D]
    w_up_exps: torch.Tensor              # [n_expert, ff, D]
    w_down_exps: torch.Tensor            # [n_expert, D, ff]

    @classmethod
    def load(cls, ts: TensorStore, i: int, dtype=torch.float32) -> "MoELayer":
        """Loads the whole MoE block. Warning: expert tensors are large
        (~500 MB each in fp32); lazy per-expert loading is a later-session
        optimization."""
        p = f"blk.{i}."

        # Expert tensors: ggml shape for gate/up_exps is (D, ff, n_expert);
        # for down_exps is (ff, D, n_expert). Transpose to (n_expert, out, in).
        g = _v(ts, p + "ffn_gate_exps.weight", dtype=dtype)  # [D, ff, n_expert]
        u = _v(ts, p + "ffn_up_exps.weight", dtype=dtype)
        d = _v(ts, p + "ffn_down_exps.weight", dtype=dtype)  # [ff, D, n_expert]
        # -> [n_expert, ff, D] for gate/up (out=ff, in=D)
        g = g.permute(2, 1, 0).contiguous()
        u = u.permute(2, 1, 0).contiguous()
        # -> [n_expert, D, ff] for down (out=D, in=ff)
        d = d.permute(2, 1, 0).contiguous()

        return cls(
            i=i,
            w_router=_w(ts, p + "ffn_gate_inp.weight", dtype=dtype),
            w_gate_sh=_w(ts, p + "ffn_gate_shexp.weight", dtype=dtype),
            w_up_sh=_w(ts, p + "ffn_up_shexp.weight", dtype=dtype),
            w_down_sh=_w(ts, p + "ffn_down_shexp.weight", dtype=dtype),
            w_gate_inp_sh=_v(ts, p + "ffn_gate_inp_shexp.weight", dtype=dtype),
            w_gate_exps=g,
            w_up_exps=u,
            w_down_exps=d,
        )


def moe_forward(x: torch.Tensor, layer: MoELayer, cfg: Config) -> torch.Tensor:
    """x: [B, T, D]. Output [B, T, D].

    Router picks top-k experts per token, softmax-normalizes their weights,
    runs each chosen expert as SwiGLU on the selected tokens, scatters the
    weighted results back. Plus an always-on shared expert.
    """
    B, T, D = x.shape
    E = cfg.n_expert
    K = cfg.n_expert_used
    x_flat = x.reshape(B * T, D)

    # --- Shared expert: SwiGLU with a scalar sigmoid gate per hidden dim.
    shgate = torch.sigmoid(x_flat * layer.w_gate_inp_sh)  # [BT, D]
    sh_in = x_flat * shgate
    sh_gate = F.linear(sh_in, layer.w_gate_sh)
    sh_up = F.linear(sh_in, layer.w_up_sh)
    sh = F.silu(sh_gate) * sh_up
    shared_out = F.linear(sh, layer.w_down_sh)             # [BT, D]

    # --- Routed experts.
    router_logits = F.linear(x_flat, layer.w_router)       # [BT, E]
    topk_vals, topk_ids = router_logits.topk(K, dim=-1)    # [BT, K]
    # Softmax over the selected top-k (the standard Qwen3/Mixtral choice).
    topk_w = F.softmax(topk_vals.float(), dim=-1).to(x.dtype)  # [BT, K]

    out_routed = torch.zeros_like(x_flat)
    for e in range(E):
        # Tokens that picked expert `e` in any of their K slots.
        mask = (topk_ids == e)
        if not mask.any():
            continue
        # Positions [n, k] where expert e was chosen.
        idx = mask.nonzero(as_tuple=False)     # [n_sel, 2]
        tok_idx = idx[:, 0]
        slot_idx = idx[:, 1]
        xe = x_flat[tok_idx]                    # [n_sel, D]
        we = topk_w[tok_idx, slot_idx]          # [n_sel]

        g = F.linear(xe, layer.w_gate_exps[e])  # [n_sel, ff]
        u = F.linear(xe, layer.w_up_exps[e])
        y = F.silu(g) * u
        y = F.linear(y, layer.w_down_exps[e])   # [n_sel, D]

        out_routed.index_add_(0, tok_idx, y * we.unsqueeze(-1))

    out = out_routed + shared_out
    return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Model-level structure
# ---------------------------------------------------------------------------

@dataclass
class Model:
    cfg: Config
    token_embd: torch.Tensor          # [vocab, D]
    output_norm: torch.Tensor         # [D]
    lm_head: torch.Tensor             # [vocab, D]
    layers: list[tuple[AttnLayer | SSMLayer, MoELayer]]
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    @classmethod
    def load(cls, ts: TensorStore, *, max_pos: int | None = None,
             dtype=torch.float32) -> "Model":
        # NOTE: eager-loads every block's MoE (256 experts × 3 mats × 40 layers
        # = ~120 GB in fp32). Will OOM on 27 GiB RAM. Next session's job:
        # swap MoELayer.load for a lazy-per-expert loader that dequants only
        # the 8 active experts per token from the mmap'd GGUF bytes.
        cfg = ts.cfg
        mp = max_pos or cfg.n_ctx_train

        # Embedding and head.
        # token_embd.weight is (D, vocab) in GGML; transpose to (vocab, D)
        # for standard F.embedding.
        emb = _w(ts, "token_embd.weight", dtype=dtype)      # [vocab, D]
        lmh = _w(ts, "output.weight", dtype=dtype)          # [vocab, D]
        onorm = _v(ts, "output_norm.weight", dtype=dtype)

        cos, sin = build_partial_rope_cache(
            cfg.rope_dim, mp, cfg.rope_theta, dtype=dtype,
        )

        layers: list[tuple[AttnLayer | SSMLayer, MoELayer]] = []
        for i in range(cfg.n_layer):
            if cfg.is_attention[i]:
                core = AttnLayer.load(ts, i)
            else:
                core = SSMLayer.load(ts, i)
            moe = MoELayer.load(ts, i, dtype=dtype)
            layers.append((core, moe))

        return cls(cfg=cfg, token_embd=emb, output_norm=onorm, lm_head=lmh,
                   layers=layers, rope_cos=cos, rope_sin=sin)


def forward(
    model: Model, tokens: Sequence[int], *,
    start_pos: int = 0,
    kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ssm_states: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, list, list]:
    """Prefill/decode forward for a single sequence.

    Returns (logits [T, vocab], kv_caches list, ssm_states list).
    """
    cfg = model.cfg
    n_layer = cfg.n_layer
    if kv_caches is None:
        kv_caches = [None] * n_layer
    if ssm_states is None:
        ssm_states = [None] * n_layer

    tok = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, T]
    x = F.embedding(tok, model.token_embd)                     # [1, T, D]

    for i, (core, moe) in enumerate(model.layers):
        if cfg.is_attention[i]:
            y, kv_caches[i] = attn_forward(
                x, core, cfg, model.rope_cos, model.rope_sin,
                kv_caches[i], start_pos,
            )
        else:
            y, ssm_states[i] = ssm_forward(x, core, cfg, ssm_states[i])
        x = x + y                                              # residual
        x = x + moe_forward(rms_norm(x, core.post_norm, cfg.rms_eps),
                            moe, cfg)                          # post-norm + MoE

    x = rms_norm(x, model.output_norm, cfg.rms_eps)
    logits = F.linear(x, model.lm_head)                        # [1, T, vocab]
    return logits.squeeze(0), kv_caches, ssm_states
