"""Measure xclbin-switching overhead.

Each NpuLinear shape has its own xclbin. Calling a mix of shapes may cost more
per call than calling one shape repeatedly (context / kernel binding flush).
This micro-bench compares:

  single-shape    : 4000 calls on ONE shape             (no switching)
  round-robin     : 1000 cycles through 4 shapes        (max switching)
  pairwise-blocks : 500 cycles of [AAAA BBBB CCCC DDDD] (mild switching)

Uses the actual SmolLM projection shapes at M_pad=256 so results are directly
comparable to the in-context profile numbers.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from npu.linear import NpuLinear


# The four fused-projection shapes from smollm.py
SHAPES = [
    ("wqkv",       576,  960),  # K, N (fused QKV: Hq*Dh + 2*Hkv*Dh = 576+192+192)
    ("wo",         576,  576),
    ("w_gate_up",  576, 3072),  # fused gate+up
    ("w_down",    1536,  576),
]
M = 256  # covers L in [1, 256] — all pad to this in the fused-proj setup


def main():
    torch.manual_seed(0)
    linears = []
    inputs = []
    for name, K, N in SHAPES:
        W = torch.randn(N, K, dtype=torch.bfloat16) / K**0.5
        lin = NpuLinear(W)
        linears.append((name, lin))
        inputs.append(torch.randn(1, M, K))

    print(f"compiling/loading xclbins for {len(SHAPES)} shapes at M={M}…")
    for (name, lin), x in zip(linears, inputs):
        y = lin(x)
        assert y.shape == (1, M, lin.out_features)
    print("all xclbins loaded.\n")

    iters = 1000

    # Round 1: single shape, 4000 calls on each — isolate per-shape kernel cost.
    print(f"{'shape':<12} {'single':>10} {'rrobin':>10} {'blocks':>10}  "
          f"{'slowdown_rr':>12} {'slowdown_blk':>12}")

    per_shape_single = {}
    for (name, lin), x in zip(linears, inputs):
        # warm
        for _ in range(20):
            lin(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            lin(x)
        dt = time.perf_counter() - t0
        per_shape_single[name] = dt / iters * 1e3

    # Round-robin: cycle A B C D A B C D ...
    # Warm
    for _ in range(10):
        for (_, lin), x in zip(linears, inputs):
            lin(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        for (_, lin), x in zip(linears, inputs):
            lin(x)
    dt_rr = time.perf_counter() - t0
    rr_ms_per_call = dt_rr / (iters * len(SHAPES)) * 1e3

    # Pairwise blocks: A A A A  B B B B  C C C C  D D D D (block_size)
    BLK = 64
    for _ in range(5):
        for (_, lin), x in zip(linears, inputs):
            for __ in range(BLK):
                lin(x)
    block_iters = max(1, iters // BLK)  # how many cycles through (4 shapes × BLK)
    t0 = time.perf_counter()
    for _ in range(block_iters):
        for (_, lin), x in zip(linears, inputs):
            for __ in range(BLK):
                lin(x)
    dt_blk = time.perf_counter() - t0
    blk_ms_per_call = dt_blk / (block_iters * len(SHAPES) * BLK) * 1e3

    # --- same-xclbin, different-weight test ---
    # Two NpuLinears with DIFFERENT weights but SAME compiled xclbin (via
    # share_N_pad). This isolates "weight-BO switch" cost from "xclbin /
    # hw_context switch" cost. Expected: if the 2.56× round-robin slowdown is
    # due to hw_context switching, this should be ~1× of single. If it's
    # driver per-call overhead that ignores xclbin identity, it stays ~2×.
    N_shared = 960
    W1 = torch.randn(N_shared, 576, dtype=torch.bfloat16) / 576**0.5
    W2 = torch.randn(N_shared, 576, dtype=torch.bfloat16) / 576**0.5
    lin_a = NpuLinear(W1, share_N_pad=N_shared)
    lin_b = NpuLinear(W2, share_N_pad=N_shared)
    x_ab = torch.randn(1, M, 576)
    # warm
    for _ in range(20):
        lin_a(x_ab); lin_b(x_ab)
    t0 = time.perf_counter()
    for _ in range(iters):
        lin_a(x_ab); lin_b(x_ab)
    dt_same_xclbin_rr = time.perf_counter() - t0
    same_xclbin_rr_ms = dt_same_xclbin_rr / (iters * 2) * 1e3

    # Also the "single" baseline at this shape for reference
    for _ in range(20):
        lin_a(x_ab)
    t0 = time.perf_counter()
    for _ in range(iters):
        lin_a(x_ab)
    same_xclbin_single_ms = (time.perf_counter() - t0) / iters * 1e3

    print(f"\nsame-xclbin, different weights:")
    print(f"  single           : {same_xclbin_single_ms:.3f} ms/call")
    print(f"  rrobin (A, B)    : {same_xclbin_rr_ms:.3f} ms/call")
    print(f"  slowdown         : {same_xclbin_rr_ms / same_xclbin_single_ms:.2f}x")
    print()

    # To compare, average per-shape single-call ms
    avg_single = sum(per_shape_single.values()) / len(per_shape_single)

    for name, _ in linears:
        s = per_shape_single[name]
        print(f"{name:<12} {s:>9.3f}ms")
    print(f"{'(mean)':<12} {avg_single:>9.3f}ms {rr_ms_per_call:>9.3f}ms {blk_ms_per_call:>9.3f}ms  "
          f"{rr_ms_per_call/avg_single:>11.2f}x {blk_ms_per_call/avg_single:>11.2f}x")
    print()
    print("Reading:")
    print("  single : avg per-call time when only ONE xclbin is in play")
    print("  rrobin : per-call time when switching every call (A B C D A B C D …)")
    print("  blocks : per-call time when running in 64-call blocks per shape")
    print("  slowdown_rr  = rrobin  / single   (switching overhead if >> 1)")
    print("  slowdown_blk = blocks  / single   (mild switching overhead if >> 1)")


if __name__ == "__main__":
    main()
