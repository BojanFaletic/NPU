"""Bring-up: NpuMatVec on the qwen router weight at T=1.

Smallest dense matvec site in the qwen forward (256 × 2048). First check
the smollm NPU stack handles a K=2048 shape; first xclbin compile is
~30 s, cached after that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "npu"))

from qwen.model import TensorStore
from npu.mv import NpuMatVec

GGUF = ROOT / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"


def main() -> int:
    print(f"loading TensorStore from {GGUF.name}")
    ts = TensorStore(GGUF)

    name = "blk.0.ffn_gate_inp.weight"
    W = torch.from_numpy(ts.get(name, dtype="fp32").copy()).to(torch.float32)
    print(f"router weight: {tuple(W.shape)}  dtype={W.dtype}")
    assert W.shape == (256, 2048), W.shape

    torch.manual_seed(0)
    x = torch.randn(2048, dtype=torch.float32) * 0.05  # roughly hidden-state scale

    y_ref = F.linear(x, W)

    print("compiling NpuMatVec (first call mints xclbin, ~30 s if cold)")
    mv = NpuMatVec(W)
    y_npu = mv(x)
    y_npu = y_npu.reshape(-1)
    assert y_npu.shape == y_ref.shape

    diff = (y_npu.float() - y_ref.float()).abs()
    cos = F.cosine_similarity(y_npu.float(), y_ref.float(), dim=0).item()
    print(
        f"router @ x : "
        f"max|d|={diff.max().item():.3e}  "
        f"mean|d|={diff.mean().item():.3e}  "
        f"cos={cos:.6f}  "
        f"|ref|={y_ref.float().norm().item():.3e}  "
        f"|npu|={y_npu.float().norm().item():.3e}"
    )
    ok = cos >= 0.999
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
