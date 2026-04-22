"""Benchmark SmolLM2 forward pass: CPU vs NPU at different prefill lengths."""
import argparse, time
import torch

from smollm import load


def bench(model, ids, warmup=2, iters=5):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.forward(ids)
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model.forward(ids)
        return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="16,64,256,512,1024")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    model_cpu, tok, _ = load(torch.float32)

    print("loading NPU model…")
    model_npu, _, _ = load(torch.float32)
    model_npu.enable_npu()
    # Warm all shapes we'll use
    print("compiling NPU xclbins…")
    for L in [int(x) for x in args.lengths.split(",")]:
        ids = torch.randint(0, 1000, (1, L))
        t0 = time.time()
        _ = model_npu.forward(ids)
        print(f"  L={L}: compile+first-call {time.time()-t0:.1f}s")

    print()
    print(f"{'length':>8}  {'cpu (ms)':>12}  {'npu (ms)':>12}  {'speedup':>8}  {'cpu tok/s':>12}  {'npu tok/s':>12}")
    for L in [int(x) for x in args.lengths.split(",")]:
        ids = torch.randint(0, 1000, (1, L))
        t_cpu = bench(model_cpu, ids, iters=args.iters)
        t_npu = bench(model_npu, ids, iters=args.iters)
        print(f"{L:>8}  {t_cpu*1e3:>12.1f}  {t_npu*1e3:>12.1f}  {t_cpu/t_npu:>8.2f}x  "
              f"{L/t_cpu:>12.1f}  {L/t_npu:>12.1f}")


if __name__ == "__main__":
    main()
