"""Bring-up: fused NpuConcatMatVec on the qwen attention Q/K/V projections.

Shapes:
  - fused wqkv : (9216, 2048)
      wq : (8192, 2048)  — gated Q: q | q_gate interleaved per head
      wk : (512, 2048)
      wv : (512, 2048)

The fused shape cold-compiles on first run. All 10 attention layers
(3, 7, 11, ..., 39) share this xclbin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from qwen.model import TensorStore
from npu.mv import NpuConcatMatVec

GGUF = ROOT / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"


def _check(name: str, y_npu: torch.Tensor, y_ref: torch.Tensor) -> bool:
    assert y_npu.shape == y_ref.shape, (y_npu.shape, y_ref.shape)
    diff = (y_npu.float() - y_ref.float()).abs()
    cos = F.cosine_similarity(y_npu.float(), y_ref.float(), dim=0).item()
    print(
        f"  [{name}] max|d|={diff.max().item():.3e}  "
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

    print("\n  [attn_qkv] W=(9216, 2048) x=(2048,)")
    print("  compiling NpuConcatMatVec (cold compile on first run; cached after)")
    mv = NpuConcatMatVec((Wq, Wk, Wv))
    y_npu = mv(x).reshape(-1)
    y_ref = torch.cat([F.linear(x, Wq), F.linear(x, Wk), F.linear(x, Wv)], dim=0)
    q_npu, k_npu, v_npu = y_npu.split((8192, 512, 512), dim=0)
    q_ref, k_ref, v_ref = y_ref.split((8192, 512, 512), dim=0)

    ok_fused = _check("attn_qkv", y_npu, y_ref)
    ok_q = _check("attn_q", q_npu, q_ref)
    ok_k = _check("attn_k", k_npu, k_ref)
    ok_v = _check("attn_v", v_npu, v_ref)

    ok = ok_fused and ok_q and ok_k and ok_v
    print("\nOK" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
