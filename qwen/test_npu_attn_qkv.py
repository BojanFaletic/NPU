"""Bring-up: NpuMatVec on the qwen attention Q/K/V projections at T=1.

Shapes:
  - wq : (8192, 2048)  — new xclbin (gated Q: q | q_gate interleaved per head)
  - wk : (512, 2048)   — shares xclbin with ffn_gate_shexp / ffn_up_shexp
  - wv : (512, 2048)   — same xclbin as wk

wq cold-compiles on first run (~30 s). wk/wv hit the cached shexp gate xclbin.
All 10 attention layers (3, 7, 11, ..., 39) share these shapes.
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

    # Layer 3 is the first attention layer.
    Wq = torch.from_numpy(ts.get("blk.3.attn_q.weight",
                                  dtype="fp32").copy()).to(torch.float32)
    Wk = torch.from_numpy(ts.get("blk.3.attn_k.weight",
                                  dtype="fp32").copy()).to(torch.float32)
    Wv = torch.from_numpy(ts.get("blk.3.attn_v.weight",
                                  dtype="fp32").copy()).to(torch.float32)
    assert Wq.shape == (8192, 2048), Wq.shape
    assert Wk.shape == (512, 2048), Wk.shape
    assert Wv.shape == (512, 2048), Wv.shape

    torch.manual_seed(0)
    # Input is post-attn_norm hidden state — O(0.05-scale) per dim.
    x = torch.randn(2048, dtype=torch.float32) * 0.05

    ok_q = _check("attn_q", x, Wq)
    ok_k = _check("attn_k", x, Wk)
    ok_v = _check("attn_v", x, Wv)

    ok = ok_q and ok_k and ok_v
    print("\nOK" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
