"""Focused test for semi-compact IQ4_XS down matvec on NPU.

Compares the packed IQ4 path against the existing host-expanded IQ4 path and a
CPU dequant + F.linear oracle for one routed expert down matrix.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npu.quant_mv_iq4 import NpuIQ4MatVec
from npu.quant_mv_iq4_packed import NpuIQ4PackedMatVec
from qwen.model import TensorStore


GGUF = ROOT / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item())


def main() -> int:
    torch.manual_seed(0)
    ts = TensorStore(GGUF)
    t = ts.raw("blk.0.ffn_down_exps.weight")
    e = 17
    K = int(t.shape[0])
    M = int(t.shape[1])
    x = torch.randn(K, dtype=torch.float32) / (K ** 0.5)

    w = torch.from_numpy(ts.get_expert(t.name, e).copy()).float()
    y_cpu = F.linear(x, w)

    t0 = time.time()
    mv_expanded = NpuIQ4MatVec.from_gguf_tensor(t, expert_idx=e)
    build_expanded = time.time() - t0
    t0 = time.time()
    y_expanded_first = mv_expanded(x)
    first_expanded = time.time() - t0
    t0 = time.time()
    y_expanded = mv_expanded(x)
    warm_expanded = time.time() - t0

    t0 = time.time()
    mv_packed = NpuIQ4PackedMatVec.from_gguf_tensor(t, expert_idx=e)
    build_packed = time.time() - t0
    t0 = time.time()
    y_packed_first = mv_packed(x)
    first_packed = time.time() - t0
    t0 = time.time()
    y_packed = mv_packed(x)
    warm_packed = time.time() - t0

    diff_packed_exp = (y_packed - y_expanded).abs()
    diff_packed_cpu = (y_packed - y_cpu).abs()
    diff_exp_cpu = (y_expanded - y_cpu).abs()
    ok = (
        cosine(y_packed, y_expanded) >= 0.99999
        and diff_packed_exp.max().item() < 5e-3
        and cosine(y_packed, y_cpu) >= 0.999
    )

    print(f"layer 0 down expert {e} IQ4_XS  M={M} K={K}")
    print(
        f"  expanded build={build_expanded*1e3:.1f}ms "
        f"first/warm={first_expanded*1e3:.1f}/{warm_expanded*1e3:.1f}ms"
    )
    print(
        f"  packed  build={build_packed*1e3:.1f}ms "
        f"first/warm={first_packed*1e3:.1f}/{warm_packed*1e3:.1f}ms"
    )
    print(
        f"  packed/expanded cos={cosine(y_packed, y_expanded):.6f} "
        f"max|d|={diff_packed_exp.max().item():.3e} "
        f"mean|d|={diff_packed_exp.mean().item():.3e}"
    )
    print(
        f"  packed/cpu      cos={cosine(y_packed, y_cpu):.6f} "
        f"max|d|={diff_packed_cpu.max().item():.3e} "
        f"mean|d|={diff_packed_cpu.mean().item():.3e}"
    )
    print(
        f"  expanded/cpu cos={cosine(y_expanded, y_cpu):.6f} "
        f"max|d|={diff_exp_cpu.max().item():.3e} "
        f"mean|d|={diff_exp_cpu.mean().item():.3e}"
    )
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
