"""Layer-by-layer isolated comparison of our forward against the llama.cpp oracle.

Each layer is fed the *oracle's* l_out-{i-1} as input (or model.input_embed
for i=0), so per-layer drift doesn't compound. We compare:
  attn_norm-i, linear_attn_out-i for the mixer
  attn_residual-i for the residual sanity
This isolates which layer has a structural bug (not just numerical drift).

Usage:
    uv run python -m qwen.compare                # all 40 layers, mixer-only
    uv run python -m qwen.compare --layers 0,1,3 # specific layers
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qwen.model import TensorStore
from qwen.forward import (
    SSMLayer, AttnLayer, MoELayer, ssm_forward, attn_forward, rms_norm,
    build_partial_rope_cache,
)


def load_oracle(d: Path, name: str) -> torch.Tensor:
    bin_p = d / f"{name}.bin"
    shp_p = d / f"{name}.shape"
    if not bin_p.exists():
        raise FileNotFoundError(bin_p)
    ne = [int(x) for x in shp_p.read_text().split()]
    arr = np.fromfile(bin_p, dtype=np.float32)
    while len(ne) > 1 and ne[-1] == 1:
        ne.pop()
    if not ne:
        ne = [arr.size]
    arr = arr.reshape(tuple(reversed(ne)))
    return torch.from_numpy(arr.copy())


def diff(name: str, ours: torch.Tensor, ref: torch.Tensor,
         tol_cos: float = 0.999) -> bool:
    o = ours.detach().float().reshape(-1)
    r = ref.detach().float().reshape(-1)
    if o.numel() != r.numel():
        print(f"  {name:<32s}  SHAPE MISMATCH ours={tuple(ours.shape)} ref={tuple(ref.shape)}")
        return False
    abs_diff = (o - r).abs()
    cos = F.cosine_similarity(o, r, dim=0).item()
    ok = cos >= tol_cos
    mark = "OK  " if ok else "FAIL"
    print(f"  {mark} {name:<28s}  "
          f"max|d|={abs_diff.max():.2e}  "
          f"cos={cos:.6f}  "
          f"|ours|={o.norm():.3e} |ref|={r.norm():.3e}")
    return ok


def compare_layer(ts: TensorStore, dump: Path, i: int,
                  cos_table: torch.Tensor, sin_table: torch.Tensor) -> None:
    cfg = ts.cfg
    is_attn = cfg.is_attention[i]
    kind = "attn" if is_attn else "ssm "
    print(f"\n=== block {i:02d} ({kind}) ===")

    # Input is oracle's previous l_out (or input_embed for i=0).
    if i == 0:
        x = load_oracle(dump, "model.input_embed")          # [D]
    else:
        x = load_oracle(dump, f"l_out-{i-1}")                # [D]
    x = x.reshape(1, 1, -1)                                  # [1,1,D]

    if is_attn:
        layer = AttnLayer.load(ts, i)
    else:
        layer = SSMLayer.load(ts, i)

    # attn_norm
    h = rms_norm(x, layer.attn_norm, cfg.rms_eps)
    diff(f"attn_norm-{i}", h.squeeze(0).squeeze(0),
         load_oracle(dump, f"attn_norm-{i}"))

    # mixer. For SSM layers ggml dumps it as `linear_attn_out-{i}`; for
    # attention layers as `attn_output-{i}` (different meaning per kind).
    if is_attn:
        y, _ = attn_forward(x, layer, cfg, cos_table, sin_table, None, 0)
        mixer_name = f"attn_output-{i}"
    else:
        y, _ = ssm_forward(x, layer, cfg, None)
        mixer_name = f"linear_attn_out-{i}"
    diff(mixer_name, y.squeeze(0).squeeze(0), load_oracle(dump, mixer_name))

    # residual (unified — exists for both layer kinds with the same meaning).
    res = (x + y).squeeze(0).squeeze(0)
    diff(f"attn_residual-{i}", res,
         load_oracle(dump, f"attn_residual-{i}"))


def chain_compare(ts: TensorStore, dump: Path,
                  cos_t: torch.Tensor, sin_t: torch.Tensor, n: int) -> None:
    """Run our forward end-to-end on the same prompt, threading our own l_out
    through every block. Compare l_out-{i} at each step to spot where drift
    compounds beyond what isolated tests would suggest."""
    cfg = ts.cfg
    print("\n=== chained forward (our l_out vs oracle) ===")

    x = load_oracle(dump, "model.input_embed").reshape(1, 1, -1)
    diff("model.input_embed", x.squeeze(0).squeeze(0),
         load_oracle(dump, "model.input_embed"))

    kv_caches = [None] * cfg.n_layer
    ssm_states = [None] * cfg.n_layer

    for i in range(n):
        is_attn = cfg.is_attention[i]
        if is_attn:
            layer = AttnLayer.load(ts, i)
            y, kv_caches[i] = attn_forward(x, layer, cfg, cos_t, sin_t,
                                            kv_caches[i], 0)
        else:
            layer = SSMLayer.load(ts, i)
            y, ssm_states[i] = ssm_forward(x, layer, cfg, ssm_states[i])
        x = x + y                                                  # attn_residual
        moe = MoELayer.load(ts, i)
        h = rms_norm(x, layer.post_norm, cfg.rms_eps)
        # Compute MoE inline (inlined moe_forward, but with diff against oracle).
        cur = h.squeeze(0).squeeze(0).unsqueeze(0)                  # [1, D]
        logits = F.linear(cur, moe.w_router).squeeze(0)
        probs = F.softmax(logits.float(), dim=-1)
        K = cfg.n_expert_used
        topk_vals, topk_ids = probs.topk(K, dim=-1)
        weights_sum = topk_vals.sum().clamp_min(1e-20)
        weights_norm = (topk_vals / weights_sum).to(cur.dtype)
        g_all, u_all, d_all = moe.dequant_experts()
        picked = topk_ids.tolist()
        moe_out_acc = torch.zeros_like(cur.squeeze(0))
        for k_idx, e in enumerate(picked):
            ge = F.linear(cur, g_all[e])
            ue = F.linear(cur, u_all[e])
            sw = F.silu(ge) * ue
            do = F.linear(sw, d_all[e]).squeeze(0)
            moe_out_acc = moe_out_acc + do * weights_norm[k_idx]
        del g_all, u_all, d_all
        # Shared expert
        sh = F.silu(F.linear(cur, moe.w_gate_sh)) * F.linear(cur, moe.w_up_sh)
        sh_out = F.linear(sh, moe.w_down_sh).squeeze(0)
        sh_gate_sig = torch.sigmoid((cur.squeeze(0) * moe.w_gate_inp_sh).sum())
        ffn_out = moe_out_acc + sh_out * sh_gate_sig
        x = x + ffn_out.reshape(1, 1, -1)
        kind = "attn" if is_attn else "ssm "
        diff(f"l_out-{i:02d} ({kind})", x.squeeze(0).squeeze(0),
             load_oracle(dump, f"l_out-{i}"))

    # Final norm + LM head
    if n == cfg.n_layer:
        ts_emb = torch.from_numpy(ts.get("output_norm.weight", dtype="fp32").copy())
        x_n = rms_norm(x, ts_emb, cfg.rms_eps).squeeze(0).squeeze(0)
        diff("result_norm", x_n, load_oracle(dump, "result_norm"))

        lm_head = torch.from_numpy(ts.get("output.weight", dtype="fp32").copy())
        logits = F.linear(x_n, lm_head)
        diff("result_output", logits, load_oracle(dump, "result_output"))

        ours_top1 = int(logits.argmax().item())
        ref_top1 = int(load_oracle(dump, "result_output").argmax().item())
        print(f"\nours_top1={ours_top1}  ref_top1={ref_top1}  "
              f"{'MATCH' if ours_top1 == ref_top1 else 'MISMATCH'}")


def compare_moe(ts: TensorStore, dump: Path, i: int, *, npu: bool = False) -> None:
    """Test MoE at layer i. Inputs come from the oracle so we isolate this
    block. Compares router logits, top-k, per-expert SwiGLU, and final ffn_out.

    With ``npu=True`` the router F.linear is routed through NpuMatVec on
    the XDNA 2 NPU; the rest of the block stays on CPU. This validates the
    NPU dispatch directly against the oracle, not just transitively.
    """
    cfg = ts.cfg
    print(f"\n=== block {i:02d} MoE{'  [npu router]' if npu else ''} ===")

    cur = load_oracle(dump, f"attn_post_norm-{i}").reshape(1, -1)   # [1, D]
    moe = MoELayer.load(ts, i)
    if npu:
        from npu.mv import NpuMatVec
        moe.npu = {"router": NpuMatVec(moe.w_router)}

    # 1) Router logits
    if npu:
        logits = moe.npu["router"](cur).squeeze(0)                  # [E]
    else:
        logits = F.linear(cur, moe.w_router).squeeze(0)             # [E]
    diff(f"ffn_moe_logits-{i}", logits, load_oracle(dump, f"ffn_moe_logits-{i}"))

    # 2) softmax over all experts
    probs = F.softmax(logits.float(), dim=-1)
    diff(f"ffn_moe_probs-{i}", probs, load_oracle(dump, f"ffn_moe_probs-{i}"))

    # 3) top-K (using probs, *not* logits — that's what ffn_moe_weights stores)
    K = cfg.n_expert_used
    topk_vals, topk_ids = probs.topk(K, dim=-1)
    diff(f"ffn_moe_weights-{i}", topk_vals, load_oracle(dump, f"ffn_moe_weights-{i}"))

    # 4) renormalize weights
    weights_sum = topk_vals.sum()
    weights_norm = topk_vals / weights_sum.clamp_min(1e-20)
    diff(f"ffn_moe_weights_norm-{i}", weights_norm,
         load_oracle(dump, f"ffn_moe_weights_norm-{i}"))

    # 5) per-expert SwiGLU. Dequant the 3 stacks once (slow but bounded).
    print(f"  dequantizing experts for layer {i} ...")
    g_all, u_all, d_all = moe.dequant_experts()           # (256,512,2048), ..., (256,2048,512)
    picked = topk_ids.tolist()
    g_stack = torch.stack([F.linear(cur.squeeze(0), g_all[e]) for e in picked])  # [K, FF]
    u_stack = torch.stack([F.linear(cur.squeeze(0), u_all[e]) for e in picked])
    sw_stack = F.silu(g_stack) * u_stack
    diff(f"ffn_moe_gate-{i}", g_stack, load_oracle(dump, f"ffn_moe_gate-{i}"))
    diff(f"ffn_moe_up-{i}", u_stack, load_oracle(dump, f"ffn_moe_up-{i}"))
    diff(f"ffn_moe_swiglu-{i}", sw_stack, load_oracle(dump, f"ffn_moe_swiglu-{i}"))

    down_stack = torch.stack([F.linear(sw_stack[k], d_all[picked[k]]) for k in range(K)])  # [K, D]
    diff(f"ffn_moe_down-{i}", down_stack, load_oracle(dump, f"ffn_moe_down-{i}"))

    weighted = down_stack * weights_norm.unsqueeze(-1).to(down_stack.dtype)  # [K, D]
    diff(f"ffn_moe_weighted-{i}", weighted, load_oracle(dump, f"ffn_moe_weighted-{i}"))

    moe_out = weighted.sum(dim=0)
    diff(f"ffn_moe_out-{i}", moe_out, load_oracle(dump, f"ffn_moe_out-{i}"))

    # 6) Shared expert
    sh_g = F.linear(cur, moe.w_gate_sh).squeeze(0)
    sh_u = F.linear(cur, moe.w_up_sh).squeeze(0)
    sh = F.silu(sh_g) * sh_u
    sh_out = F.linear(sh, moe.w_down_sh)
    diff(f"ffn_shexp-{i}", sh_out, load_oracle(dump, f"ffn_shexp-{i}"))

    # 7) Shared gate (1-D dot product → scalar)
    shared_gate = (cur.squeeze(0) * moe.w_gate_inp_sh).sum()
    diff(f"shared_expert_gate-{i}", shared_gate.unsqueeze(0),
         load_oracle(dump, f"shared_expert_gate-{i}"))
    shared_gate_sig = torch.sigmoid(shared_gate)
    diff(f"shared_expert_gate_sigmoid-{i}", shared_gate_sig.unsqueeze(0),
         load_oracle(dump, f"shared_expert_gate_sigmoid-{i}"))

    sh_gated = sh_out * shared_gate_sig
    diff(f"ffn_shexp_gated-{i}", sh_gated, load_oracle(dump, f"ffn_shexp_gated-{i}"))

    # 8) Final FFN
    ffn_out = moe_out + sh_gated
    diff(f"ffn_out-{i}", ffn_out, load_oracle(dump, f"ffn_out-{i}"))

    # 9) full block: l_out = attn_residual + ffn_out
    l_out = load_oracle(dump, f"attn_residual-{i}") + ffn_out
    diff(f"l_out-{i}", l_out, load_oracle(dump, f"l_out-{i}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf")
    ap.add_argument("--dump", default="/tmp/qwen_oracle_1tok")
    ap.add_argument("--layers", default=None,
                    help="comma-sep layer indices; default = 0..n_layer-1")
    ap.add_argument("--moe-layer", type=int, default=None,
                    help="run MoE comparison for this single layer")
    ap.add_argument("--chain", action="store_true",
                    help="chain our forward through layers (uses our outputs, "
                         "not oracle's) and compare l_out at every step")
    ap.add_argument("--chain-n", type=int, default=40,
                    help="how many layers to chain (default: all 40)")
    ap.add_argument("--npu", action="store_true",
                    help="route NPU-eligible ops (currently: router) through "
                         "XDNA 2 NpuMatVec instead of F.linear; only meaningful "
                         "with --moe-layer.")
    args = ap.parse_args()

    dump = Path(args.dump)
    ts = TensorStore(args.model)
    cfg = ts.cfg

    cos, sin = build_partial_rope_cache(cfg.rope_dim, 512, cfg.rope_theta)

    if args.moe_layer is not None:
        compare_moe(ts, dump, args.moe_layer, npu=args.npu)
        return 0

    if args.chain:
        chain_compare(ts, dump, cos, sin, args.chain_n)
        return 0

    if args.layers is None:
        layers = range(cfg.n_layer)
    else:
        layers = [int(x) for x in args.layers.split(",")]

    for i in layers:
        compare_layer(ts, dump, i, cos, sin)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
