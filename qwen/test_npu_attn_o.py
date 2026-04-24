"""Bring-up: NpuMatVec on the qwen attention output projection at T=1.

Single new shape (2048, 4096). All 10 attention layers (3, 7, 11, …, 39)
share the same shape, so this test validates the kernel for every one
of them. First xclbin compile is ~30 s, cached after that.
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


def main() -> int:
    print(f"loading TensorStore from {GGUF.name}")
    ts = TensorStore(GGUF)

    name = "blk.3.attn_output.weight"
    W = torch.from_numpy(ts.get(name, dtype="fp32").copy()).to(torch.float32)
    print(f"attn_output weight: {tuple(W.shape)}  dtype={W.dtype}")
    assert W.shape == (2048, 4096), W.shape

    torch.manual_seed(0)
    # post-gated-attention input scale: gating × post-softmax(v) is roughly
    # O(0.1) per dim, so this is in-range without manufacturing tail-mass.
    x = torch.randn(4096, dtype=torch.float32) * 0.1

    print("compiling NpuMatVec (cold compile ~30 s; cached after)")
    mv = NpuMatVec(W)
    y_ref = F.linear(x, W)
    y_npu = mv(x).reshape(-1)
    assert y_npu.shape == y_ref.shape

    diff = (y_npu.float() - y_ref.float()).abs()
    cos = F.cosine_similarity(y_npu.float(), y_ref.float(), dim=0).item()
    print(
        f"attn_output @ x: "
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
