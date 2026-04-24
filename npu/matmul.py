"""bf16 matmul on XDNA 2 (AIE2p / Krackan).

End-to-end pipeline, self-contained:
  1. Compile the stock mm.cc from mlir-aie's aie_kernels/aie2p with Peano.
  2. Generate IRON MLIR orchestrating tiled matmul on one compute tile.
  3. aiecc -> xclbin + insts.bin
  4. pyxrt dispatch on /dev/accel/accel0
  5. Validate vs torch.matmul (bf16 in -> fp32 out)
"""
from __future__ import annotations
import argparse, os, subprocess, sys, shutil
from pathlib import Path

import numpy as np
import torch

# -------------------- toolchain paths --------------------

VENV_SITE = Path(__file__).parent.parent / ".venv" / "lib" / "python3.12" / "site-packages"
PEANO    = VENV_SITE / "llvm-aie"
MLIR_AIE = VENV_SITE / "mlir_aie"
KERNEL_SRC = Path(__file__).parent.parent / "vendor" / "mlir-aie-src" / "aie_kernels" / "aie2p"


def compile_kernel(m: int, k: int, n: int, build_dir: Path) -> Path:
    """Peano-compile aie_kernels/aie2p/mm.cc -> mm_mxkxn.o."""
    obj = build_dir / f"mm_{m}x{k}x{n}.o"
    src = KERNEL_SRC / "mm.cc"
    include = MLIR_AIE / "include"
    clang = PEANO / "bin" / "clang++"
    cmd = [
        str(clang),
        "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf", "-DNDEBUG",
        "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
        "-Wno-empty-body", "-Wno-missing-template-arg-list-after-template-kw",
        "-I", str(include),
        f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}",
        "-Dbf16_f32_ONLY",  # select only bf16-in / f32-out variant
        "-c", str(src), "-o", str(obj),
    ]
    print(f"[peano] compiling mm_{m}x{k}x{n}.o")
    subprocess.run(cmd, check=True, cwd=build_dir)
    return obj


# -------------------- IRON MLIR generation --------------------

def generate_mlir(M: int, K: int, N: int, m: int, k: int, n: int, out_path: Path) -> None:
    """Single-core bf16-in / f32-out tiled matmul MLIR (npu2/aie2p)."""
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2
    from aie.iron.placers import SequentialPlacer
    from aie.helpers.taplib import TensorTiler2D

    dtype_in  = np.dtype(np.float32).newbyteorder("=")  # placeholder
    # IRON expects np.dtype; bf16 is exposed via aie.iron.str_to_dtype
    from aie.iron import str_to_dtype
    dt_in  = str_to_dtype("bf16")
    dt_out = str_to_dtype("f32")

    # mac vector dims for aie2p bf16 (non-emulated path)
    r, s, t = 4, 8, 8
    assert m % r == 0 and k % s == 0 and n % t == 0

    A_ty = np.ndarray[(M * K,), np.dtype[dt_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dt_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dt_out]]
    a_ty = np.ndarray[(m, k), np.dtype[dt_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dt_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dt_out]]

    # Kernel function declarations pointing to the compiled .o
    obj_name = f"mm_{m}x{k}x{n}.o"
    zero_k   = Kernel("zero_f32",      obj_name, [c_ty])
    matmul_k = Kernel("matmul_bf16_f32", obj_name, [a_ty, b_ty, c_ty])

    # Data-movement fabric — stock tiling from single_core_iron.py (vectorized)
    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_ty, name="inB")
    b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    tiles = M_div_m * N_div_n

    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_c = of_c.acquire(1)
            zero(elem_c)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                ea = of_a.acquire(1)
                eb = of_b.acquire(1)
                matmul(ea, eb, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_k, matmul_k],
        stack_size=0xD00,
    )

    # Tile access patterns
    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n, prune_step=False
    )
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K_div_k, N_div_n),
        tile_group_col_major=True, prune_step=False,
    )[0]
    rows_per_block = 4
    C_tiles = TensorTiler2D.group_tiler(
        (M, N), (m, n), (rows_per_block // 2, N_div_n), prune_step=False
    )
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        tgs = []

        def ceildiv(a, b): return (a + b - 1) // b

        for blk in range(ceildiv(M_div_m, rows_per_block)):
            for pp in [0, 1]:
                row_base = blk * rows_per_block + pp * rows_per_block // 2
                num_rows = min(rows_per_block // 2, M_div_m - row_base)
                if num_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tr in range(num_rows):
                    off = (row_base + tr) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[off], task_group=tgs[-1])
                    rt.fill(inB.prod(), B, tap=b_tap,       task_group=tgs[-1])
                rt.drain(outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True)
                c_index += 1
                if blk > 0 or (blk == 0 and pp > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    prog = Program(NPU2(), rt)
    module = prog.resolve_program(SequentialPlacer())
    out_path.write_text(str(module))


# -------------------- compile MLIR -> xclbin --------------------

def compile_xclbin(mlir: Path, obj: Path, build: Path) -> tuple[Path, Path]:
    xclbin = build / "final.xclbin"
    insts  = build / "insts.bin"
    # aiecc expects the .o in the cwd so MLIR symbol "mm_MxKxN.o" resolves
    assert obj.parent == build
    aiecc = shutil.which("aiecc") or str(MLIR_AIE / "bin" / "aiecc")
    cmd = [
        aiecc, "--aie-generate-xclbin", f"--xclbin-name={xclbin.name}",
        "--no-xchesscc", "--no-xbridge", f"--peano={PEANO}",
        "--aie-generate-npu-insts", f"--npu-insts-name={insts.name}",
        str(mlir.resolve()),
    ]
    print("[aiecc]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=build)
    return xclbin, insts


# -------------------- dispatch via pyxrt + verify --------------------

def _bf16_to_u16(t: torch.Tensor) -> np.ndarray:
    # torch bf16 tensor -> uint16 numpy with same bit pattern
    return t.contiguous().view(torch.uint16).numpy()


def run_and_check(xclbin_path: Path, insts_path: Path, M: int, K: int, N: int) -> None:
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt

    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(str(xclbin_path))
    kname = next(k.get_name() for k in xclbin.get_kernels() if k.get_name().startswith("MLIR_AIE"))
    uuid = device.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(device, uuid)
    kernel = pyxrt.kernel(ctx, kname)

    insts = np.fromfile(insts_path, dtype=np.uint32)
    bo_instr = pyxrt.bo(device, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    bo_instr.write(insts.tobytes(), 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)
    C_ref = (A.float() @ B.float())  # fp32 reference

    a_np = _bf16_to_u16(A).reshape(-1)
    b_np = _bf16_to_u16(B).reshape(-1)
    c_np = np.zeros(M * N, dtype=np.float32)

    bo_a = pyxrt.bo(device, a_np.nbytes, pyxrt.bo.host_only, kernel.group_id(3))
    bo_b = pyxrt.bo(device, b_np.nbytes, pyxrt.bo.host_only, kernel.group_id(4))
    bo_c = pyxrt.bo(device, c_np.nbytes, pyxrt.bo.host_only, kernel.group_id(5))

    bo_a.write(a_np.tobytes(), 0); bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_b.write(b_np.tobytes(), 0); bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_c.write(c_np.tobytes(), 0); bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    run = kernel(3, bo_instr, insts.size, bo_a, bo_b, bo_c)
    run.wait()

    bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    c_out = np.frombuffer(bytes(bo_c.read(c_np.nbytes, 0)), dtype=np.float32).reshape(M, N)
    C_got = torch.from_numpy(c_out)

    absdiff = (C_got - C_ref).abs()
    reldiff = absdiff / (C_ref.abs() + 1e-6)
    print(f"shape=({M}x{K})·({K}x{N})  max|Δ|={absdiff.max():.3e}  "
          f"mean|Δ|={absdiff.mean():.3e}  max rel={reldiff.max():.3e}")

    # bf16 gives ~3 decimal digits; for random matrices, K contributions can grow.
    tol = 3e-2 * (K ** 0.5)  # loose, scales with sqrt(K)
    assert absdiff.max() < tol, f"mismatch exceeds tol {tol:.3f}"
    print("OK")


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=64)
    ap.add_argument("-K", type=int, default=64)
    ap.add_argument("-N", type=int, default=64)
    ap.add_argument("-m", type=int, default=32)
    ap.add_argument("-k", type=int, default=32)
    ap.add_argument("-n", type=int, default=32)
    args = ap.parse_args()

    root = Path(__file__).parent
    build = root / "build" / f"mm_{args.M}x{args.K}x{args.N}_{args.m}x{args.k}x{args.n}"
    build.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] peano compile kernel -> {build}")
    obj = compile_kernel(args.m, args.k, args.n, build)

    mlir = build / "aie.mlir"
    print(f"[2/4] generating IRON MLIR -> {mlir}")
    generate_mlir(args.M, args.K, args.N, args.m, args.k, args.n, mlir)

    print(f"[3/4] aiecc -> xclbin")
    xclbin, insts = compile_xclbin(mlir, obj, build)

    print(f"[4/4] dispatching and verifying")
    run_and_check(xclbin, insts, args.M, args.K, args.N)


if __name__ == "__main__":
    main()
