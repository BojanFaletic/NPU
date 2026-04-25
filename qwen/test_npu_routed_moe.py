"""Validate the T=1 routed MoE NPU path for one Qwen layer.

This compares the normal CPU expert path against `moe_forward` with lazy NPU
expert handles enabled. It exercises top-k routing, IQ3_XXS gate/up experts,
IQ4_XS down experts, SwiGLU, route weighting, and the shared expert add.

Run:
    uv run python qwen/test_npu_routed_moe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwen.forward import MoELayer, moe_forward
from qwen.model import TensorStore


GGUF = ROOT / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    return float((af @ bf) / (af.norm() * bf.norm()).clamp_min(1e-20))


def main() -> int:
    ts = TensorStore(GGUF)
    moe = MoELayer.load(ts, 0)

    torch.manual_seed(0)
    x = torch.randn(1, 1, ts.cfg.d_model, dtype=torch.float32) * 0.05

    t0 = time.time()
    ref = moe_forward(x, moe, ts.cfg)
    t_cpu = time.time() - t0

    p = "blk.0."
    moe.npu = {
        "expert_tensors": (
            ts.raw(p + "ffn_gate_exps.weight"),
            ts.raw(p + "ffn_up_exps.weight"),
            ts.raw(p + "ffn_down_exps.weight"),
        ),
        "expert_cache": {},
    }

    t0 = time.time()
    out = moe_forward(x, moe, ts.cfg)
    t_first = time.time() - t0
    t0 = time.time()
    out = moe_forward(x, moe, ts.cfg)
    t_warm = time.time() - t0

    c = cos(out, ref)
    max_err = (out - ref).abs().max().item()
    rel_err = max_err / ref.abs().max().item()
    ok = c >= 0.999
    print(
        f"layer 0 routed MoE T=1\n"
        f"  cpu: {t_cpu*1000:.1f}ms   "
        f"npu first/warm: {t_first*1000:.1f}/{t_warm*1000:.1f}ms   "
        f"cached experts={len(moe.npu['expert_cache'])}\n"
        f"  cos={c:.6f}  max|d|={max_err:.3e}  rel={rel_err:.3e}  "
        f"{'OK' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
