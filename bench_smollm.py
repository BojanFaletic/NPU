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
    lengths = [int(x) for x in args.lengths.split(",")]

    model_cpu, tok, _ = load(torch.float32)

    print("loading NPU (proj only) model…")
    model_npu_proj, _, _ = load(torch.float32)
    model_npu_proj.enable_npu(softmax=False)

    print("loading NPU (proj + softmax) model…")
    model_npu_full, _, _ = load(torch.float32)
    model_npu_full.enable_npu(softmax=True)

    print("compiling NPU xclbins…")
    for L in lengths:
        ids = torch.randint(0, 1000, (1, L))
        t0 = time.time()
        _ = model_npu_proj.forward(ids)
        _ = model_npu_full.forward(ids)
        print(f"  L={L}: compile+first-call {time.time()-t0:.1f}s")

    print()
    print(f"{'length':>8}  {'cpu':>10}  {'npu-proj':>10}  {'npu-full':>10}  "
          f"{'p-speedup':>10}  {'f-speedup':>10}  {'sm-delta':>10}")
    for L in lengths:
        ids = torch.randint(0, 1000, (1, L))
        t_cpu  = bench(model_cpu,       ids, iters=args.iters)
        t_proj = bench(model_npu_proj,  ids, iters=args.iters)
        t_full = bench(model_npu_full,  ids, iters=args.iters)
        print(f"{L:>8}  {t_cpu*1e3:>9.1f}ms  {t_proj*1e3:>9.1f}ms  {t_full*1e3:>9.1f}ms  "
              f"{t_cpu/t_proj:>9.2f}x  {t_cpu/t_full:>9.2f}x  "
              f"{(t_full-t_proj)*1e3:>+9.1f}ms")


if __name__ == "__main__":
    main()
