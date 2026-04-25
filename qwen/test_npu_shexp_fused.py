"""Validate the fused T=1 shared-expert NPU path for one Qwen layer.

This exercises the single-dispatch gate/up/SwiGLU/down kernel used by the
``shexp`` enable_npu option.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npu.mlp import NpuFusedMLP
from qwen.model import TensorStore


GGUF = ROOT / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    return float((af @ bf) / (af.norm() * bf.norm()).clamp_min(1e-20))


def staged_ref(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    x_bf = x.to(torch.bfloat16).float()
    wg = w_gate.to(torch.bfloat16).float()
    wu = w_up.to(torch.bfloat16).float()
    wd = w_down.to(torch.bfloat16).float()
    gate = F.linear(x_bf, wg)
    up = F.linear(x_bf, wu)
    act = (F.silu(gate) * up).to(torch.bfloat16).float()
    return F.linear(act, wd)


def main() -> int:
    ts = TensorStore(GGUF)
    Wg = torch.from_numpy(ts.get("blk.0.ffn_gate_shexp.weight", dtype="fp32").copy())
    Wu = torch.from_numpy(ts.get("blk.0.ffn_up_shexp.weight", dtype="fp32").copy())
    Wd = torch.from_numpy(ts.get("blk.0.ffn_down_shexp.weight", dtype="fp32").copy())
    assert Wg.shape == (512, 2048), Wg.shape
    assert Wu.shape == (512, 2048), Wu.shape
    assert Wd.shape == (2048, 512), Wd.shape

    torch.manual_seed(0)
    x = torch.randn(1, 2048, dtype=torch.float32) * 0.05

    ref_stage = staged_ref(x, Wg, Wu, Wd)
    ref_fp32 = F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)

    mlp = NpuFusedMLP(Wg, Wd, w_up=Wu)
    t0 = time.time()
    y = mlp(x)
    t_first = time.time() - t0
    t0 = time.time()
    y2 = mlp(x)
    t_warm = time.time() - t0

    c_stage = cos(y, ref_stage)
    c_fp32 = cos(y, ref_fp32)
    max_stage = (y - ref_stage).abs().max().item()
    max_repeat = (y2 - y).abs().max().item()
    ok = c_stage >= 0.999 and c_fp32 >= 0.999 and max_repeat == 0.0
    print(
        f"layer 0 fused shared expert T=1\n"
        f"  first/warm: {t_first*1000:.1f}/{t_warm*1000:.1f}ms\n"
        f"  cos staged/fp32={c_stage:.6f}/{c_fp32:.6f}  "
        f"max|d staged|={max_stage:.3e}  repeat={max_repeat:.3e}  "
        f"{'OK' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
