"""Bring-up: NpuMatVec on the qwen shared-expert weights at T=1.

Two unique shapes:
  - gate / up : (512, 2048)  — one xclbin, gate and up share it
  - down      : (2048, 512)  — second xclbin

First xclbin compile is ~30 s, cached after that. Run on layer 0; all 40
shared experts share the same shapes, so this validates the kernel for
every layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from qwen.model import TensorStore
from npu.mv import NpuMatVec

GGUF = ROOT / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"


def _check(name: str, x: torch.Tensor, W: torch.Tensor) -> bool:
    print(f"\n  [{name}]  W={tuple(W.shape)}  x={tuple(x.shape)}")
    print("  compiling NpuMatVec (cold compile ~30 s; cached after)")
    mv = NpuMatVec(W)
    y_ref = F.linear(x, W)
    y_npu = mv(x).reshape(-1)
    assert y_npu.shape == y_ref.shape, (y_npu.shape, y_ref.shape)
    diff = (y_npu.float() - y_ref.float()).abs()
    cos = F.cosine_similarity(y_npu.float(), y_ref.float(), dim=0).item()
    print(
        f"  max|d|={diff.max().item():.3e}  "
        f"mean|d|={diff.mean().item():.3e}  "
        f"cos={cos:.6f}  "
        f"|ref|={y_ref.float().norm().item():.3e}  "
        f"|npu|={y_npu.float().norm().item():.3e}"
    )
    return cos >= 0.999


def main() -> int:
    print(f"loading TensorStore from {GGUF.name}")
    ts = TensorStore(GGUF)

    Wg = torch.from_numpy(ts.get("blk.0.ffn_gate_shexp.weight",
                                  dtype="fp32").copy()).to(torch.float32)
    Wu = torch.from_numpy(ts.get("blk.0.ffn_up_shexp.weight",
                                  dtype="fp32").copy()).to(torch.float32)
    Wd = torch.from_numpy(ts.get("blk.0.ffn_down_shexp.weight",
                                  dtype="fp32").copy()).to(torch.float32)
    assert Wg.shape == (512, 2048), Wg.shape
    assert Wu.shape == (512, 2048), Wu.shape
    assert Wd.shape == (2048, 512), Wd.shape

    torch.manual_seed(0)
    x_in = torch.randn(2048, dtype=torch.float32) * 0.05    # post_norm-scale
    x_mid = torch.randn(512, dtype=torch.float32) * 0.5     # silu(g)*u scale

    ok_g = _check("ffn_gate_shexp", x_in, Wg)
    ok_u = _check("ffn_up_shexp",   x_in, Wu)
    ok_d = _check("ffn_down_shexp", x_mid, Wd)

    ok = ok_g and ok_u and ok_d
    print("\nOK" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
