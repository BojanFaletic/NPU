"""Bring-up: NpuMatVec on the qwen SSM (Gated DeltaNet) projections at T=1.

Shapes:
  - w_in  (attn_qkv) : (8192, 2048) — shares xclbin with attn wq
  - w_gate (attn_gate): (4096, 2048) — new xclbin
  - w_out (ssm_out)  : (2048, 4096) — shares xclbin with attn_output

alpha/beta (32, 2048) are intentionally skipped — they're tiny enough
that NPU dispatch overhead dominates. All 30 SSM layers share these
shapes, so this validates the kernel for every one.
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

    # Layer 0 is an SSM layer (attention layers are at 3, 7, 11, ...).
    Wi = torch.from_numpy(ts.get("blk.0.attn_qkv.weight",
                                  dtype="fp32").copy()).to(torch.float32)
    Wg = torch.from_numpy(ts.get("blk.0.attn_gate.weight",
                                  dtype="fp32").copy()).to(torch.float32)
    Wo = torch.from_numpy(ts.get("blk.0.ssm_out.weight",
                                  dtype="fp32").copy()).to(torch.float32)
    assert Wi.shape == (8192, 2048), Wi.shape
    assert Wg.shape == (4096, 2048), Wg.shape
    assert Wo.shape == (2048, 4096), Wo.shape

    torch.manual_seed(0)
    # w_in / w_gate take post-attn_norm hidden state — O(0.05-scale).
    x_in = torch.randn(2048, dtype=torch.float32) * 0.05
    # w_out takes silu(z)*rms_norm(o) — O(0.1) per dim.
    x_out = torch.randn(4096, dtype=torch.float32) * 0.1

    ok_i = _check("ssm_in",   x_in,  Wi)
    ok_g = _check("ssm_gate", x_in,  Wg)
    ok_o = _check("ssm_out",  x_out, Wo)

    ok = ok_i and ok_g and ok_o
    print("\nOK" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
