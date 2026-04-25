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

import os
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from qwen.model import Config, TensorStore, from_bf16

try:
    from npu.profiler import profile
except ModuleNotFoundError:
    def profile(_name: str):
        return nullcontext()


# ---------------------------------------------------------------------------
# Weight loading: GGML-order numpy → torch. GGUF stores weights in
# (in, out) order; torch's F.linear expects (out, in). We transpose on load.
# ---------------------------------------------------------------------------

def _t(ts: TensorStore, name: str, dtype=torch.float32) -> torch.Tensor:
    """Load a tensor as a torch Tensor in reversed(ggml shape) layout — which
    for standard Linear weights means (out, in) directly, no transpose."""
    arr_fp32 = ts.get(name, dtype="fp32", keep=False)
    return torch.from_numpy(arr_fp32.copy()).to(dtype)


_w = _t  # weights: same as _t — the reshape in TensorStore already gives (out, in).
_v = _t  # 1-D vectors / norms: same load path.


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
    # NPU dispatch handles for the wqkv / wo F.linear sites,
    # populated by enable_npu(). Each entry is callable equivalent to
    # F.linear(x, W). T=1 path uses NpuMatVec; T>1 falls back to F.linear.
    npu: dict | None = None

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

    with profile("attn.norm"):
        h = rms_norm(x, layer.attn_norm, cfg.rms_eps)

    # Gated Q with *interleaved per-head* layout:
    #   [q_h0 | gate_h0 | q_h1 | gate_h1 | ... | q_h{Hq-1} | gate_h{Hq-1}]
    # (llama.cpp qwen35moe.cpp uses stride n_embd_head*2 per head; gate view
    # is offset by n_embd_head within each slot.)
    npu_qkv = (
        layer.npu is not None
        and ("wqkv" in layer.npu or "wq" in layer.npu)
        and B == 1 and T == 1
    )
    with profile("attn.qkv"):
        if npu_qkv:
            if "wqkv" in layer.npu:
                qkv = layer.npu["wqkv"](h.reshape(-1))
                q_full, k_raw, v_raw = qkv.split(
                    (2 * Hq * Dh, Hkv * Dh, Hkv * Dh), dim=-1,
                )
                q_full = q_full.reshape(1, 1, 2 * Hq * Dh)
                k_raw = k_raw.reshape(1, 1, Hkv * Dh)
                v_raw = v_raw.reshape(1, 1, Hkv * Dh)
            else:
                q_full = layer.npu["wq"](h.reshape(-1)).reshape(1, 1, 2 * Hq * Dh)
                k_raw = layer.npu["wk"](h.reshape(-1)).reshape(1, 1, Hkv * Dh)
                v_raw = layer.npu["wv"](h.reshape(-1)).reshape(1, 1, Hkv * Dh)
        else:
            q_full = F.linear(h, layer.wq)           # [B, T, 2*Hq*Dh]
            k_raw = F.linear(h, layer.wk)
            v_raw = F.linear(h, layer.wv)

    with profile("attn.core"):
        q_full = q_full.view(B, T, Hq, 2, Dh)
        q = q_full[..., 0, :]                     # [B, T, Hq, Dh]
        q_gate = q_full[..., 1, :]                # [B, T, Hq, Dh]
        k = k_raw.view(B, T, Hkv, Dh)
        v = v_raw.view(B, T, Hkv, Dh)

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

    with profile("attn.out_proj"):
        if (
            layer.npu is not None
            and "wo" in layer.npu
            and B == 1 and T == 1
        ):
            out = layer.npu["wo"](out.reshape(-1)).reshape(1, 1, D)
        else:
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
    # NPU dispatch handles (populated by enable_npu()) for w_in / w_gate /
    # w_out at T=1. alpha/beta are 32-row — too small to be worth dispatching.
    npu: dict | None = None

    @classmethod
    def load(cls, ts: TensorStore, i: int) -> "SSMLayer":
        p = f"blk.{i}."
        # With the reversed-reshape convention, ssm_conv1d comes out (8192, 4)
        # directly — (channel, kernel) which F.conv1d's depthwise layout wants.
        return cls(
            i=i,
            attn_norm=_v(ts, p + "attn_norm.weight"),
            post_norm=_v(ts, p + "post_attention_norm.weight"),
            w_in=_w(ts, p + "attn_qkv.weight"),
            w_gate=_w(ts, p + "attn_gate.weight"),
            conv1d=_v(ts, p + "ssm_conv1d.weight"),
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

    # attn_norm applies to SSM input too (same as for attention layers — in
    # qwen35moe.cpp it's applied *outside* the mixer call for both branches).
    with profile("ssm.norm"):
        h = rms_norm(x, layer.attn_norm, cfg.rms_eps)

    # --- Input projections ---------------------------------------------------
    npu_ssm = (
        layer.npu is not None
        and "w_in" in layer.npu
        and B == 1 and T == 1
    )
    with profile("ssm.in_proj"):
        if npu_ssm:
            qkv_mixed = layer.npu["w_in"](h.reshape(-1)).reshape(1, 1, conv_ch)
            z = layer.npu["w_gate"](h.reshape(-1)).reshape(1, 1, d_inner)
        else:
            qkv_mixed = F.linear(h, layer.w_in)          # [B, T, 8192]
            z = F.linear(h, layer.w_gate)                # [B, T, 4096]

    # --- α, β per head per token --------------------------------------------
    # alpha_proj → (B, T, Hv); add dt_bias; softplus; multiply by a_coef.
    with profile("ssm.alpha_beta"):
        alpha_raw = F.linear(h, layer.alpha)          # [B, T, Hv]
        beta_raw = F.linear(h, layer.beta)            # [B, T, Hv]
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

    with profile("ssm.conv"):
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

    with profile("ssm.recurrence"):
        # --- Split into q, k, v --------------------------------------------------
        q = conv_out[..., :Ck].view(B, T, Hk, Sk)               # [B, T, Hk, Sk]
        k = conv_out[..., Ck:2 * Ck].view(B, T, Hk, Sk)
        v = conv_out[..., 2 * Ck:].view(B, T, Hv, Sv)           # [B, T, Hv, Sv]

        # L2-normalize q, k per-head.
        q = _l2_norm(q, cfg.rms_eps)
        k = _l2_norm(k, cfg.rms_eps)

        # Broadcast K heads to V heads (Hk -> Hv, repeat factor `rep`). llama.cpp
        # uses ggml_repeat_4d here, which produces a *tile* pattern
        # [h0, h1, ..., h15, h0, h1, ..., h15] — so V head i pairs with K head
        # (i mod Hk). Use torch .repeat (tile), not .repeat_interleave (block).
        q = q.repeat(1, 1, rep, 1)                              # [B, T, Hv, Sk]
        k = k.repeat(1, 1, rep, 1)

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

    with profile("ssm.gated_norm"):
        # --- Gated norm: rmsnorm(o, ssm_norm) * silu(z) -------------------------
        # ssm_norm applies over the last dim (Sv=128) per head.
        o = rms_norm(o, layer.ssm_norm, cfg.rms_eps)             # [B, T, Hv, Sv]
        o = o.reshape(B, T, Hv * Sv)                             # [B, T, 4096]
        o = o * F.silu(z)                                        # [B, T, 4096]

    # --- Output projection ---------------------------------------------------
    with profile("ssm.out_proj"):
        if npu_ssm:
            out = layer.npu["w_out"](o.reshape(-1)).reshape(1, 1, D)
        else:
            out = F.linear(o, layer.w_out)                       # [B, T, D]

    return out, (new_conv_state, S)


# ---------------------------------------------------------------------------
# MoE layer
# ---------------------------------------------------------------------------

@dataclass
class MoELayer:
    """Lazy MoE layer. Small weights (router + shared expert) are resident;
    the 256 × 3 expert matrices are re-dequantized from the mmap'd GGUF on
    every moe_forward call. That's ~7–8 s per call on IQ3_XXS/IQ4_XS but
    avoids a 120 GB RAM eager load."""
    i: int
    ts: TensorStore = field(repr=False)
    dtype: torch.dtype = torch.float32
    # Router + shared expert (kept resident — small, ~20 MB per layer).
    w_router: torch.Tensor | None = None          # [n_expert, D]
    w_gate_sh: torch.Tensor | None = None          # [ff_sh, D]
    w_up_sh: torch.Tensor | None = None
    w_down_sh: torch.Tensor | None = None          # [D, ff_sh]
    w_gate_inp_sh: torch.Tensor | None = None      # [D]
    # NPU dispatch handles, populated by enable_npu(). Dense entries are
    # callables equivalent to F.linear(x, W); "sh_mlp" is equivalent to the
    # full shared expert MLP before its scalar gate. T=1 paths use the NPU;
    # T>1 falls back to torch since the decode kernels are single-vector only.
    npu: dict | None = None

    @classmethod
    def load(cls, ts: TensorStore, i: int, dtype=torch.float32) -> "MoELayer":
        p = f"blk.{i}."
        return cls(
            i=i, ts=ts, dtype=dtype,
            w_router=_w(ts, p + "ffn_gate_inp.weight", dtype=dtype),
            w_gate_sh=_w(ts, p + "ffn_gate_shexp.weight", dtype=dtype),
            w_up_sh=_w(ts, p + "ffn_up_shexp.weight", dtype=dtype),
            w_down_sh=_w(ts, p + "ffn_down_shexp.weight", dtype=dtype),
            w_gate_inp_sh=_v(ts, p + "ffn_gate_inp_shexp.weight", dtype=dtype),
        )

    def dequant_experts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dequantize all 3 stacked expert tensors at once. Slow (~9 s/layer)
        and only used by tests. The hot path goes through dequant_one()."""
        p = f"blk.{self.i}."
        g = _w(self.ts, p + "ffn_gate_exps.weight", dtype=self.dtype)
        u = _w(self.ts, p + "ffn_up_exps.weight", dtype=self.dtype)
        d = _w(self.ts, p + "ffn_down_exps.weight", dtype=self.dtype)
        return g, u, d

    def dequant_one(self, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dequantize *one* expert's gate/up/down — ~36 ms for IQ3_XXS vs
        ~9 s for the full stack. Returns (gate[ff,D], up[ff,D], down[D,ff])."""
        p = f"blk.{self.i}."
        g = self.ts.get_expert(p + "ffn_gate_exps.weight", expert_idx)
        u = self.ts.get_expert(p + "ffn_up_exps.weight", expert_idx)
        d = self.ts.get_expert(p + "ffn_down_exps.weight", expert_idx)
        return (
            torch.from_numpy(g.copy()).to(self.dtype),
            torch.from_numpy(u.copy()).to(self.dtype),
            torch.from_numpy(d.copy()).to(self.dtype),
        )


def _npu_routed_expert(layer: MoELayer, expert_idx: int):
    """Per-expert dispatch handles. Most tensors are IQ3_XXS (gate/up) or
    IQ4_XS (down) and run on NPU. A few outliers in blk.1 / blk.34 are Q5_K /
    Q6_K — for those we keep a per-expert CPU dequant + F.linear closure so
    the routed path stays uniform from moe_forward's perspective.
    """
    cache = layer.npu["expert_cache"]
    cached = cache.get(expert_idx)
    if cached is not None:
        if hasattr(cache, "move_to_end"):
            cache.move_to_end(expert_idx)
        return cached

    from gguf import GGMLQuantizationType
    from npu.mv import NpuMatVec
    from npu.quant_mv import (
        BLK as IQ3_BLK,
        B_BYTES as IQ3_B_BYTES,
        NpuQuantMatVec,
        repack_iq3_xxs_per_expert,
    )
    from npu.quant_mv_iq4 import NpuIQ4MatVec
    from npu.quant_mv_iq4_packed import NpuIQ4PackedMatVec

    gate_t, up_t, down_t = layer.npu["expert_tensors"]

    def _make(t, *, packed_iq4: bool = False):
        if layer.npu.get("expert_dense"):
            w_np = layer.ts.get_expert(t.name, expert_idx)
            w = torch.from_numpy(w_np.astype(np.float32, copy=True))
            return NpuMatVec(w)
        qt = t.tensor_type
        if qt == GGMLQuantizationType.IQ3_XXS:
            return NpuQuantMatVec.from_gguf_tensor(t, expert_idx=expert_idx)
        if qt == GGMLQuantizationType.IQ4_XS:
            if packed_iq4 and layer.npu.get("expert_packed_down"):
                return NpuIQ4PackedMatVec.from_gguf_tensor(t, expert_idx=expert_idx)
            return NpuIQ4MatVec.from_gguf_tensor(t, expert_idx=expert_idx)
        w_np = layer.ts.get_expert(t.name, expert_idx)
        w = torch.from_numpy(w_np.astype(np.float32, copy=False))
        return lambda x: F.linear(x, w)

    if (
        layer.npu.get("expert_fused")
        and not layer.npu.get("expert_dense")
        and gate_t.tensor_type == GGMLQuantizationType.IQ3_XXS
        and up_t.tensor_type == GGMLQuantizationType.IQ3_XXS
    ):
        K_in = int(gate_t.shape[0])
        ffn = int(gate_t.shape[1])
        row_bytes = (K_in // IQ3_BLK) * IQ3_B_BYTES
        gate = repack_iq3_xxs_per_expert(gate_t, expert_idx).reshape(ffn, row_bytes)
        up = repack_iq3_xxs_per_expert(up_t, expert_idx).reshape(ffn, row_bytes)
        gate_up = np.ascontiguousarray(np.concatenate([gate, up], axis=0)).reshape(-1)
        cached = (
            NpuQuantMatVec.from_iq3_xxs_bytes(gate_up, 2 * ffn, K_in),
            _make(down_t, packed_iq4=True),
        )
    else:
        cached = (_make(gate_t), _make(up_t), _make(down_t, packed_iq4=True))
    cache[expert_idx] = cached
    limit = layer.npu.get("expert_cache_limit")
    if limit is not None:
        while len(cache) > limit:
            _, evicted = cache.popitem(last=False)
            del evicted
    return cached


def moe_forward(x: torch.Tensor, layer: MoELayer, cfg: Config) -> torch.Tensor:
    """x: [B, T, D]. Output [B, T, D]. Dequants the 3 expert tensors for this
    layer once per call (slow: ~7 s), drops them on return."""
    B, T, D = x.shape
    E = cfg.n_expert
    K = cfg.n_expert_used
    x_flat = x.reshape(B * T, D)

    # --- Shared expert: SwiGLU followed by a *scalar* (per-token) sigmoid gate.
    # ffn_gate_inp_shexp is shape (D,): llama.cpp's build_lora_mm treats the
    # 1-D weight as a dot product, producing one gate value per token.
    npu_sh_mlp = (
        layer.npu is not None
        and "sh_mlp" in layer.npu
        and x_flat.shape[0] == 1
    )
    npu_shexp_split = (
        layer.npu is not None
        and "sh_gate" in layer.npu
        and x_flat.shape[0] == 1
    )
    with profile("moe.shared"):
        if npu_sh_mlp:
            shared_out = layer.npu["sh_mlp"](x_flat)             # [1, D]
        else:
            if npu_shexp_split:
                sh_gate = layer.npu["sh_gate"](x_flat)
                sh_up = layer.npu["sh_up"](x_flat)
            else:
                sh_gate = F.linear(x_flat, layer.w_gate_sh)
                sh_up = F.linear(x_flat, layer.w_up_sh)
            sh = F.silu(sh_gate) * sh_up
            if npu_shexp_split:
                shared_out = layer.npu["sh_down"](sh)            # [1, D]
            else:
                shared_out = F.linear(sh, layer.w_down_sh)       # [BT, D]
        shared_scalar_gate = torch.sigmoid(x_flat @ layer.w_gate_inp_sh)  # [BT]
        shared_out = shared_out * shared_scalar_gate.unsqueeze(-1)

    # --- Routed experts. Match llama.cpp's qwen35moe ordering: softmax over
    # all E experts first, then top-K, then renormalize the top-K weights.
    with profile("moe.router_topk"):
        if layer.npu is not None and "router" in layer.npu and x_flat.shape[0] == 1:
            router_logits = layer.npu["router"](x_flat)               # [1, E]
        else:
            router_logits = F.linear(x_flat, layer.w_router)          # [BT, E]
        probs = F.softmax(router_logits.float(), dim=-1).to(x.dtype)  # [BT, E]
        topk_vals, topk_ids = probs.topk(K, dim=-1)                   # [BT, K]
        topk_w = topk_vals / topk_vals.sum(-1, keepdim=True).clamp_min(1e-20)
        picked = torch.unique(topk_ids)

    out_routed = torch.zeros_like(x_flat)
    npu_experts = (
        layer.npu is not None
        and "expert_tensors" in layer.npu
        and x_flat.shape[0] == 1
    )
    # Dequantize only the experts that at least one token picked. With T=1
    # that's exactly K=8; with longer prefills it stays bounded by min(K*T, E).
    with profile("moe.routed_loop"):
        for e in picked.tolist():
            mask = (topk_ids == e)
            idx = mask.nonzero(as_tuple=False)
            tok_idx = idx[:, 0]
            slot_idx = idx[:, 1]
            xe = x_flat[tok_idx]
            we = topk_w[tok_idx, slot_idx]

            if npu_experts:
                with profile("moe.expert.resolve"):
                    expert = _npu_routed_expert(layer, int(e))
                if len(expert) == 2:
                    gate_up_mv, down_mv = expert
                    with profile("moe.expert.gate_up"):
                        gu = gate_up_mv(xe)
                    with profile("moe.expert.act"):
                        g, u = gu.split(cfg.d_expert_ff, dim=-1)
                        y = F.silu(g) * u
                    with profile("moe.expert.down"):
                        y = down_mv(y)
                else:
                    gate_mv, up_mv, down_mv = expert
                    with profile("moe.expert.gate_up"):
                        g = gate_mv(xe)
                        u = up_mv(xe)
                    with profile("moe.expert.act"):
                        y = F.silu(g) * u
                    with profile("moe.expert.down"):
                        y = down_mv(y)
            else:
                with profile("moe.expert.cpu_dequant"):
                    wg, wu, wd = layer.dequant_one(int(e))
                with profile("moe.expert.cpu_linear"):
                    g = F.linear(xe, wg)
                    u = F.linear(xe, wu)
                    y = F.silu(g) * u
                    y = F.linear(y, wd)
                del wg, wu, wd

            with profile("moe.expert.accum"):
                out_routed.index_add_(0, tok_idx, y * we.unsqueeze(-1))

    return (out_routed + shared_out).view(B, T, D)


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


def _expert_cache_limit_from_env(default: int) -> int | None:
    raw = os.environ.get("QWEN_NPU_EXPERT_CACHE_LIMIT")
    if raw is None or raw == "":
        return default
    val = raw.strip().lower()
    if val in {"none", "unlimited", "inf", "-1"}:
        return None
    limit = int(val)
    return None if limit < 0 else limit


def enable_npu(
    model: Model,
    ops: Sequence[str] = ("router",),
    *,
    expert_cache_limit: int | None = None,
) -> None:
    """Wrap selected dense F.linear sites in NpuMatVec on the XDNA 2 NPU.

    Currently supported ops:
        router    — MoE router (40 layers, [n_expert=256, D=2048])
        shexp     — fused shared expert gate/up/activation/down in one
                    dispatch per layer at T=1
        shexp_split — legacy shared expert gate/up/down as three dense
                    matvec dispatches per layer at T=1
        attn_o    — attention output proj (10 attn layers, [2048, 4096])
        attn_qkv  — fused attention Q/K/V projections (10 attn layers;
                    [9216, 2048] = Q [8192, 2048] + K/V [512, 2048])
        ssm       — Gated DeltaNet in/gate/out (30 SSM layers; w_in shares
                    [8192, 2048] with attn wq, w_out shares [2048, 4096]
                    with attn_o, w_gate [4096, 2048] is new)
        experts   — routed MoE experts at T=1 with fused IQ3_XXS gate/up
                    in one NPU dispatch, followed by the IQ4_XS down dispatch.
        experts_packed_down — use a semi-compact IQ4_XS down format
                    (int8 qvalues + fp32 group scales) to reduce routed down
                    expert cache memory and first-use repack time.
        experts_split — legacy routed MoE experts at T=1 as separate gate/up/
                    down NPU dispatches. Experts are materialized lazily as
                    routing selects them.
        experts_dense — routed MoE experts at T=1, but each selected expert
                    is dequantized to bf16 once and then uses the fast dense
                    NPU matvec path. Higher memory and first-use cost, much
                    faster once the selected experts are cached.

    Routed expert caches are per-layer LRU caches. By default compact and dense
    experts keep 8 experts/layer; override with
    `expert_cache_limit` or QWEN_NPU_EXPERT_CACHE_LIMIT. A negative value or
    "unlimited" disables eviction.

    Each new (out, in) shape mints one xclbin (~30 s cold, cached after).
    Decode kernels are T=1-only; the dispatch path falls back to torch for T>1.
    """
    from npu.mlp import NpuFusedMLP
    from npu.mv import NpuConcatMatVec, NpuMatVec

    valid = {
        "router", "shexp", "shexp_split", "attn_o", "attn_qkv", "ssm",
        "experts", "experts_split", "experts_dense", "experts_packed_down",
    }
    unknown = set(ops) - valid
    if unknown:
        raise ValueError(f"unknown NPU ops: {sorted(unknown)} (valid: {sorted(valid)})")

    for core, moe in model.layers:
        if moe.npu is None:
            moe.npu = {}
        if "router" in ops:
            # All routers share shape [n_expert=256, D=2048] → one xclbin total.
            moe.npu["router"] = NpuMatVec(moe.w_router)
        if "shexp" in ops:
            moe.npu["sh_mlp"] = NpuFusedMLP(
                moe.w_gate_sh, moe.w_down_sh, w_up=moe.w_up_sh,
            )
        if "shexp_split" in ops:
            # gate/up share (512, 2048); down is (2048, 512). Two xclbins total.
            moe.npu["sh_gate"] = NpuMatVec(moe.w_gate_sh)
            moe.npu["sh_up"] = NpuMatVec(moe.w_up_sh)
            moe.npu["sh_down"] = NpuMatVec(moe.w_down_sh)
        if (
            "experts" in ops
            or "experts_split" in ops
            or "experts_dense" in ops
            or "experts_packed_down" in ops
        ):
            p = f"blk.{moe.i}."
            expert_dense = "experts_dense" in ops
            expert_fused = "experts" in ops and "experts_split" not in ops and not expert_dense
            if expert_cache_limit is None:
                cache_limit = _expert_cache_limit_from_env(8)
            else:
                cache_limit = None if expert_cache_limit < 0 else expert_cache_limit
            moe.npu["expert_tensors"] = (
                moe.ts.raw(p + "ffn_gate_exps.weight"),
                moe.ts.raw(p + "ffn_up_exps.weight"),
                moe.ts.raw(p + "ffn_down_exps.weight"),
            )
            moe.npu["expert_cache"] = OrderedDict()
            moe.npu["expert_dense"] = expert_dense
            moe.npu["expert_fused"] = expert_fused
            moe.npu["expert_packed_down"] = "experts_packed_down" in ops
            moe.npu["expert_cache_limit"] = cache_limit
        if "attn_o" in ops and isinstance(core, AttnLayer):
            # All 10 attention layers share shape (2048, 4096) → one xclbin.
            if core.npu is None:
                core.npu = {}
            core.npu["wo"] = NpuMatVec(core.wo)
        if "attn_qkv" in ops and isinstance(core, AttnLayer):
            # One fused Q/K/V dispatch. The wrapper stages the three source
            # tensors into one XRT buffer without materialising torch.cat.
            if core.npu is None:
                core.npu = {}
            core.npu["wqkv"] = NpuConcatMatVec(
                (core.wq, core.wk, core.wv), profile_prefix="attn_qkv_mv",
            )
        if "ssm" in ops and isinstance(core, SSMLayer):
            if core.npu is None:
                core.npu = {}
            core.npu["w_in"] = NpuMatVec(core.w_in)
            core.npu["w_gate"] = NpuMatVec(core.w_gate)
            core.npu["w_out"] = NpuMatVec(core.w_out)


def forward(
    model: Model, tokens: Sequence[int], *,
    start_pos: int = 0,
    kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ssm_states: list[torch.Tensor | None] | None = None,
    n_layer: int | None = None,
    skip_moe: bool = False,
    trace: bool = False,
) -> tuple[torch.Tensor, list, list]:
    """Prefill/decode forward for a single sequence.

    Debug knobs: `n_layer` runs only the first N layers (for fast iteration),
    `skip_moe` replaces the MoE with zeros (skips the 7 s/layer dequant),
    `trace` prints per-layer hidden-state norm so we can see where a run
    diverges from sanity (NaN/inf blowup).
    """
    cfg = model.cfg
    total_layers = n_layer if n_layer is not None else cfg.n_layer
    if kv_caches is None:
        kv_caches = [None] * cfg.n_layer
    if ssm_states is None:
        ssm_states = [None] * cfg.n_layer

    with profile("forward.embed"):
        tok = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, T]
        x = F.embedding(tok, model.token_embd)                     # [1, T, D]
    if trace:
        print(f"    emb: norm={x.float().norm().item():.3e}  "
              f"max={x.float().abs().max().item():.3e}")

    for i in range(total_layers):
        core, moe = model.layers[i]
        kind = "attn" if cfg.is_attention[i] else "ssm "
        if cfg.is_attention[i]:
            with profile("forward.attn"):
                y, kv_caches[i] = attn_forward(
                    x, core, cfg, model.rope_cos, model.rope_sin,
                    kv_caches[i], start_pos,
                )
        else:
            with profile("forward.ssm"):
                y, ssm_states[i] = ssm_forward(x, core, cfg, ssm_states[i])
        x = x + y
        if skip_moe:
            moe_out = torch.zeros_like(x)
        else:
            with profile("forward.moe_norm"):
                moe_in = rms_norm(x, core.post_norm, cfg.rms_eps)
            with profile("forward.moe"):
                moe_out = moe_forward(moe_in, moe, cfg)
        x = x + moe_out
        if trace:
            print(f"    l{i:02d} {kind}: norm={x.float().norm().item():.3e}  "
                  f"max={x.float().abs().max().item():.3e}")

    with profile("forward.out_norm"):
        x = rms_norm(x, model.output_norm, cfg.rms_eps)
    with profile("forward.lm_head"):
        logits = F.linear(x, model.lm_head)                        # [1, T, vocab]
    return logits.squeeze(0), kv_caches, ssm_states
