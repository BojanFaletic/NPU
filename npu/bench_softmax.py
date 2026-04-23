"""Throughput bench: NPU softmax vs torch CPU softmax at attention-realistic shapes.

Shapes mirror SmolLM2 attention during prefill:
    rows = B * Hq * T = 1 * 9 * T
    L    = Tk = T (first-prefill; no past cache)
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from softmax import NpuSoftmax


def bench_once(rows: int, L: int, iters: int = 50, warmup: int = 5):
    torch.manual_seed(0)
    x = torch.randn(rows, L, dtype=torch.bfloat16)

    # --- CPU reference: match what smollm.py does on the CPU path ---
    # `F.softmax(att.float(), dim=-1).to(bf16)`
    def cpu_run():
        return torch.softmax(x.float(), dim=-1).to(torch.bfloat16)

    for _ in range(warmup): cpu_run()
    t0 = time.perf_counter()
    for _ in range(iters): cpu_run()
    cpu_ms = (time.perf_counter() - t0) / iters * 1000

    # --- NPU path ---
    sm = NpuSoftmax(n_cores=1)
    for _ in range(warmup): sm(x)  # triggers compile + buffer alloc
    t0 = time.perf_counter()
    for _ in range(iters): sm(x)
    npu_ms = (time.perf_counter() - t0) / iters * 1000

    print(f"  rows={rows:5d}  L={L:4d}  cpu={cpu_ms:7.3f} ms  npu={npu_ms:7.3f} ms  "
          f"ratio = {cpu_ms/npu_ms:.2f}x (>1 means NPU wins)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    # realistic prefill shapes for SmolLM2 (B=1, Hq=9)
    shapes = [
        (1  * 9 * 16,  16),
        (1  * 9 * 32,  32),
        (1  * 9 * 64,  64),
        (1  * 9 * 128, 128),
        (1  * 9 * 256, 256),
        (1  * 9 * 512, 512),
    ]
    print("Softmax throughput bench (rows, L):")
    for rows, L in shapes:
        bench_once(rows, L, iters=args.iters)


if __name__ == "__main__":
    main()
