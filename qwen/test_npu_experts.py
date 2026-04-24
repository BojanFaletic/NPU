"""Per-expert routed-MoE matvec on NPU via NpuQuantMatVec (IQ3_XXS dequant on
tile). Validates that NPU dequant+matvec matches gguf-py dequant + F.linear to
cosine ≥ 0.999 across multiple shapes/experts.

Run:
    uv run python qwen/test_npu_experts.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import IQ3_XXS as GGUF_IQ3

from npu.quant_mv import NpuQuantMatVec, repack_iq3_xxs_per_expert


GGUF = ROOT / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    return float((af @ bf) / (af.norm() * bf.norm()).clamp_min(1e-20))


def cpu_dequant_then_linear(t, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
    chunk = np.ascontiguousarray(t.data[expert_idx]).view(np.uint8)
    full = GGUF_IQ3.dequantize_blocks(chunk.reshape(-1, 98))
    M = int(t.shape[1])
    K = int(t.shape[0])
    W = torch.from_numpy(full.reshape(M, K).astype(np.float32))
    return F.linear(x, W)


def main() -> int:
    r = GGUFReader(str(GGUF))
    by_name = {t.name: t for t in r.tensors}

    # gate_exps: shape (in=2048, out=512, n_expert=256). Per-expert M=512, K=2048.
    # down_exps: shape (in=512, out=2048, n_expert=256). Per-expert M=2048, K=512.
    cases = [
        ("blk.0.ffn_gate_exps.weight", 17),  # IQ3_XXS, K=2048, M=512
        ("blk.0.ffn_up_exps.weight",   17),  # IQ3_XXS, K=2048, M=512
        ("blk.0.ffn_down_exps.weight", 17),  # IQ4_XS — will fall back to gguf-py
        ("blk.5.ffn_gate_exps.weight",  3),  # IQ3_XXS, different expert idx
    ]

    torch.manual_seed(0)
    rc = 0
    for name, e in cases:
        t = by_name[name]
        if t.tensor_type != GGMLQuantizationType.IQ3_XXS:
            print(f"\n{name} expert {e}: skipping ({t.tensor_type.name})")
            continue
        K = int(t.shape[0])
        M = int(t.shape[1])
        x = torch.randn(K, dtype=torch.float32) * 0.05
        ref = cpu_dequant_then_linear(t, e, x).reshape(-1)

        t0 = time.time()
        mv = NpuQuantMatVec.from_gguf_tensor(t, expert_idx=e)
        t_build = time.time() - t0
        t0 = time.time()
        out = mv(x).reshape(-1)
        t_dispatch = time.time() - t0

        c = cos(out, ref)
        max_err = (out - ref).abs().max().item()
        rel_err = max_err / ref.abs().max().item()
        ok = c >= 0.999
        rc |= 0 if ok else 1
        print(
            f"\n{name} expert {e}  M={M} K={K}\n"
            f"  build/cache: {t_build:.1f}s   dispatch: {t_dispatch*1000:.1f}ms\n"
            f"  cos={c:.6f}  max|Δ|={max_err:.3e}  rel={rel_err:.3e}  "
            f"{'OK' if ok else 'FAIL'}"
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
