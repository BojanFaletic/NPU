"""Quick throughput benchmark for our bf16 matmul.

Compiles the kernel once, then runs it in a tight loop and reports TFLOPS.
Compares against torch bf16 matmul on CPU.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from matmul import compile_kernel, generate_mlir, compile_xclbin, _bf16_to_u16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=64)
    ap.add_argument("-K", type=int, default=576)
    ap.add_argument("-N", type=int, default=576)
    ap.add_argument("-m", type=int, default=32)
    ap.add_argument("-k", type=int, default=32)
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    root = Path(__file__).parent
    build = root / "build" / f"mm_{args.M}x{args.K}x{args.N}_{args.m}x{args.k}x{args.n}"
    build.mkdir(parents=True, exist_ok=True)

    # Build (cached if already present)
    xclbin = build / "final.xclbin"
    insts  = build / "insts.bin"
    if not xclbin.exists() or not insts.exists():
        print("[build] compiling kernel + xclbin...")
        obj = compile_kernel(args.m, args.k, args.n, build)
        mlir = build / "aie.mlir"
        generate_mlir(args.M, args.K, args.N, args.m, args.k, args.n, mlir)
        compile_xclbin(mlir, obj, build)

    # Set up XRT + kernel once
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt
    device = pyxrt.device(0)
    xb = pyxrt.xclbin(str(xclbin))
    kname = next(k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE"))
    uuid = device.register_xclbin(xb)
    ctx = pyxrt.hw_context(device, uuid)
    kernel = pyxrt.kernel(ctx, kname)

    inst_arr = np.fromfile(insts, dtype=np.uint32)
    bo_instr = pyxrt.bo(device, inst_arr.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    bo_instr.write(inst_arr.tobytes(), 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    M, K, N = args.M, args.K, args.N
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)

    a_np = _bf16_to_u16(A).reshape(-1)
    b_np = _bf16_to_u16(B).reshape(-1)

    bo_a = pyxrt.bo(device, a_np.nbytes, pyxrt.bo.host_only, kernel.group_id(3))
    bo_b = pyxrt.bo(device, b_np.nbytes, pyxrt.bo.host_only, kernel.group_id(4))
    bo_c = pyxrt.bo(device, M * N * 4, pyxrt.bo.host_only, kernel.group_id(5))
    bo_a.write(a_np.tobytes(), 0); bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_b.write(b_np.tobytes(), 0); bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Warmup
    for _ in range(args.warmup):
        run = kernel(3, bo_instr, inst_arr.size, bo_a, bo_b, bo_c)
        run.wait()

    # Timed: just the dispatch+wait (device time, incl. DMA of inputs already resident)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        run = kernel(3, bo_instr, inst_arr.size, bo_a, bo_b, bo_c)
        run.wait()
    bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    t_npu = (time.perf_counter() - t0) / args.iters

    flops = 2 * M * K * N
    print(f"[NPU ] ({M}x{K})·({K}x{N})  {t_npu*1e6:8.1f} µs/iter   {flops/t_npu/1e9:7.2f} GFLOPS")

    # CPU reference: torch bf16 on CPU
    for _ in range(args.warmup):
        _ = A.float() @ B.float()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = A.float() @ B.float()
    t_cpu = (time.perf_counter() - t0) / args.iters
    print(f"[CPU ] ({M}x{K})·({K}x{N})  {t_cpu*1e6:8.1f} µs/iter   {flops/t_cpu/1e9:7.2f} GFLOPS (fp32)")

    print(f"  speedup NPU/CPU: {t_cpu/t_npu:.2f}x")


if __name__ == "__main__":
    main()
