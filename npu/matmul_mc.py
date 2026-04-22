"""Multi-core bf16 matmul on XDNA 2 — uses `whole_array_iron` across n_aie_cols columns.

Reuses the stock `my_matmul()` from the mlir-aie example rather than
rewriting the fabric.  We wrap it with compile + dispatch + verify.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from matmul import compile_kernel, compile_xclbin, _bf16_to_u16, PEANO, MLIR_AIE

# Make the example's `my_matmul` importable
EX = Path(__file__).parent.parent / "vendor" / "mlir-aie-src" / "programming_examples" / "basic" / "matrix_multiplication" / "whole_array"
sys.path.insert(0, str(EX))
import whole_array_iron as wai


def generate_mlir(M, K, N, m, k, n, n_cols, out_path: Path):
    module = wai.my_matmul(
        dev="npu2", M=M, K=K, N=N, m=m, k=k, n=n,
        n_aie_cols=n_cols,
        dtype_in_str="bf16", dtype_out_str="f32",
        b_col_maj=0, emulate_bf16_mmul_with_bfp16=False,
        trace_size=0, generate_taps=False,
    )
    out_path.write_text(str(module))


def run_and_verify(xclbin_path: Path, insts_path: Path, M: int, K: int, N: int, iters: int, warmup: int) -> None:
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt

    dev = pyxrt.device(0)
    xb  = pyxrt.xclbin(str(xclbin_path))
    kname = next(k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE"))
    uuid = dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, uuid)
    kernel = pyxrt.kernel(ctx, kname)

    insts = np.fromfile(insts_path, dtype=np.uint32)
    bo_i = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    bo_i.write(insts.tobytes(), 0); bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)
    C_ref = A.float() @ B.float()

    a = _bf16_to_u16(A).reshape(-1)
    b = _bf16_to_u16(B).reshape(-1)
    bo_a = pyxrt.bo(dev, a.nbytes,       pyxrt.bo.host_only, kernel.group_id(3))
    bo_b = pyxrt.bo(dev, b.nbytes,       pyxrt.bo.host_only, kernel.group_id(4))
    bo_c = pyxrt.bo(dev, M*N*4,          pyxrt.bo.host_only, kernel.group_id(5))
    bo_a.write(a.tobytes(), 0); bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_b.write(b.tobytes(), 0); bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # one run for correctness
    run = kernel(3, bo_i, insts.size, bo_a, bo_b, bo_c); run.wait()
    bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    c_out = np.frombuffer(bytes(bo_c.read(M*N*4, 0)), dtype=np.float32).reshape(M, N)
    diff = (torch.from_numpy(c_out) - C_ref).abs()
    print(f"correctness: max|Δ|={diff.max():.3e}  mean|Δ|={diff.mean():.3e}")

    # throughput loop
    for _ in range(warmup):
        kernel(3, bo_i, insts.size, bo_a, bo_b, bo_c).wait()
    t0 = time.perf_counter()
    for _ in range(iters):
        kernel(3, bo_i, insts.size, bo_a, bo_b, bo_c).wait()
    bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    t = (time.perf_counter() - t0) / iters
    flops = 2 * M * K * N
    print(f"perf   : ({M}x{K})·({K}x{N})  {t*1e6:8.1f} µs/iter   {flops/t/1e9:7.2f} GFLOPS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=512)
    ap.add_argument("-K", type=int, default=576)
    ap.add_argument("-N", type=int, default=576)
    ap.add_argument("-m", type=int, default=32)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--cols", type=int, default=4, choices=[1, 2, 4, 8])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    build = Path(__file__).parent / "build" / f"mm_mc_{args.M}x{args.K}x{args.N}_{args.m}x{args.k}x{args.n}_c{args.cols}"
    build.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] peano compile kernel (m={args.m} k={args.k} n={args.n}) -> {build}")
    obj = compile_kernel(args.m, args.k, args.n, build)

    mlir = build / "aie.mlir"
    print(f"[2/4] generating IRON MLIR ({args.cols} cols)")
    generate_mlir(args.M, args.K, args.N, args.m, args.k, args.n, args.cols, mlir)

    print(f"[3/4] aiecc -> xclbin")
    xclbin, insts = compile_xclbin(mlir, obj, build)

    print(f"[4/4] dispatch + bench")
    run_and_verify(xclbin, insts, args.M, args.K, args.N, args.iters, args.warmup)


if __name__ == "__main__":
    main()
