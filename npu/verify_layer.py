"""Verify that replacing F.linear(h, W) with NpuLinear(W)(h) inside one SmolLM2
layer produces near-identical output vs the pure-CPU forward.

This is the bridge test: if the hidden state after layer 0 matches when we swap
the Q-projection for the NPU version, then our NpuLinear contract is correct
and we can plug it into the full forward pass confidently.
"""
import sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from smollm import Config, Layer, SmolLM, load, rms_norm, apply_rope

sys.path.insert(0, str(Path(__file__).parent))
from linear import NpuLinear


def run_layer_cpu(layer: Layer, x, cos, sin, cfg: Config):
    B, T, D = x.shape
    Hq, Hkv, Dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    h = rms_norm(x, layer.ln1, cfg.rms_eps)
    q = F.linear(h, layer.wq).view(B, T, Hq,  Dh).transpose(1, 2)
    k = F.linear(h, layer.wk).view(B, T, Hkv, Dh).transpose(1, 2)
    v = F.linear(h, layer.wv).view(B, T, Hkv, Dh).transpose(1, 2)
    return q, k, v, h


def run_layer_npu(layer: Layer, x, cos, sin, cfg: Config, npu_wq: NpuLinear):
    B, T, D = x.shape
    Hq, Hkv, Dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    h = rms_norm(x, layer.ln1, cfg.rms_eps)
    # NPU returns fp32; rest of pipeline also fp32; cast q through view/transpose.
    q_flat = npu_wq(h)                             # [B, T, Hq*Dh] fp32
    q = q_flat.view(B, T, Hq, Dh).transpose(1, 2)
    k = F.linear(h, layer.wk).view(B, T, Hkv, Dh).transpose(1, 2)
    v = F.linear(h, layer.wv).view(B, T, Hkv, Dh).transpose(1, 2)
    return q, k, v, h


def main():
    model, tok, _ = load(torch.float32)
    ids = tok("Once upon a time, there", return_tensors="pt").input_ids
    T = ids.shape[1]
    print(f"prompt tokens = {T}")

    # Run embedding + layer 0 on CPU to get input x
    x = model.embed[ids]
    layer0 = model.layers[0]

    # Build NpuLinear for Q projection (in=576, out=576)
    print("building NpuLinear for Q proj …")
    t0 = time.time()
    npu_wq = NpuLinear(layer0.wq)
    # warm: trigger xclbin build for T=current
    _ = npu_wq(x)
    print(f"  build+first-call took {time.time()-t0:.1f}s")

    with torch.no_grad():
        q_cpu, k_cpu, v_cpu, h_cpu = run_layer_cpu(layer0, x, model.cos, model.sin, model.cfg)
        q_npu, k_npu, v_npu, h_npu = run_layer_npu(layer0, x, model.cos, model.sin, model.cfg, npu_wq)

    # Sanity: h, k, v should be bit-identical (CPU path, untouched)
    for name, a, b in [("h", h_cpu, h_npu), ("k", k_cpu, k_npu), ("v", v_cpu, v_npu)]:
        assert torch.equal(a, b), f"{name} diverged — bug in test scaffolding"

    d = (q_cpu - q_npu).abs()
    scale = q_cpu.abs().max().item()
    print(f"Q-projection output   : shape={tuple(q_cpu.shape)}  max|Δ|={d.max():.3e}  "
          f"mean|Δ|={d.mean():.3e}  scale={scale:.3f}  relΔ={d.max()/scale:.3e}")
    # bf16 is ~3 decimal digits; sum of K=576 contributions can reach ~1% of output scale
    assert d.max() / scale < 1e-2, "Q projection rel-error too large"
    print("OK — NPU Q-projection matches CPU within bf16 rounding")


if __name__ == "__main__":
    main()
