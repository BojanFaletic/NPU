"""Hello world for XDNA2 NPU.

Pipeline:
  1. Define a passthrough kernel in IRON (external DDR -> compute tile -> DDR).
  2. aiecc compiles the IRON Python to an xclbin + NPU instruction stream.
  3. aie.xrt loads the xclbin on /dev/accel/accel0 and runs it.
  4. Host verifies output == input.

Success = first user-scheduled dispatch on Krackan Point.
"""
from __future__ import annotations
import os, sys, subprocess, shutil
from pathlib import Path

import numpy as np


def generate_mlir(length: int, out_path: Path) -> None:
    """Emit AIE MLIR for a length-N int32 passthrough via compute tile DMA."""
    # Build the IRON program. Use NPU2Col1 — single column on aie2p (Strix/Krackan).
    from aie.iron import ObjectFifo, Program, Runtime
    from aie.iron.device import NPU2Col1
    from aie.iron.placers import SequentialPlacer

    line = 1024
    assert length % line == 0, "length must be multiple of line size"
    vector_ty = np.ndarray[(length,), np.dtype[np.int32]]
    line_ty   = np.ndarray[(line,),   np.dtype[np.int32]]

    of_in  = ObjectFifo(line_ty, name="in")
    of_out = of_in.cons().forward()  # consumer-side compute tile forwards to shim

    rt = Runtime()
    with rt.sequence(vector_ty, vector_ty, vector_ty) as (a_in, _b, c_out):
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    prog = Program(NPU2Col1(), rt)
    module = prog.resolve_program(SequentialPlacer())
    out_path.write_text(str(module))


def compile_xclbin(mlir_path: Path, build_dir: Path) -> tuple[Path, Path]:
    """Run aiecc to produce an xclbin and insts.bin."""
    build_dir.mkdir(parents=True, exist_ok=True)
    xclbin = build_dir / "final.xclbin"
    insts  = build_dir / "insts.bin"
    cmd = [
        "aiecc",
        "--aie-generate-xclbin", f"--xclbin-name={xclbin.name}",
        "--no-xchesscc", "--no-xbridge",
        "--aie-generate-npu-insts", f"--npu-insts-name={insts.name}",
        str(mlir_path.resolve()),
    ]
    print(f"[aiecc] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=build_dir, check=True)
    return xclbin, insts


def run_on_device(xclbin_path: Path, insts_path: Path, length: int) -> None:
    # The aie.xrt binding is hard-pinned to RyzenAI-Phoenix, so we use the
    # XRT-shipped pyxrt module directly.
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt

    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(str(xclbin_path))
    kernels = xclbin.get_kernels()
    # IRON names the only kernel "MLIR_AIE" (possibly with a suffix).
    kernel_name = next(k.get_name() for k in kernels if k.get_name().startswith("MLIR_AIE"))
    uuid = device.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(device, uuid)
    kernel = pyxrt.kernel(ctx, kernel_name)

    # Load NPU instruction stream into a CACHEABLE buffer, arg slot 1.
    insts = np.fromfile(insts_path, dtype=np.uint32)
    bo_instr = pyxrt.bo(device, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    bo_instr.write(insts.tobytes(), 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Three HOST_ONLY data buffers at arg slots 3, 4, 5 — matches IRON "runtime
    # sequence (a_in, _b, c_out)" signature + opcode at 0 + instr_buf at 1/2.
    size_bytes = length * np.dtype(np.int32).itemsize
    bo_in  = pyxrt.bo(device, size_bytes, pyxrt.bo.host_only, kernel.group_id(3))
    bo_mid = pyxrt.bo(device, size_bytes, pyxrt.bo.host_only, kernel.group_id(4))
    bo_out = pyxrt.bo(device, size_bytes, pyxrt.bo.host_only, kernel.group_id(5))

    src = np.arange(length, dtype=np.int32)
    bo_in.write(src.tobytes(), 0)
    bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    opcode = 3  # "execute instruction buffer"
    run = kernel(opcode, bo_instr, insts.size, bo_in, bo_mid, bo_out)
    run.wait()

    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out_bytes = bo_out.read(size_bytes, 0)
    dst = np.frombuffer(bytes(out_bytes), dtype=np.int32)

    mism = int((src != dst).sum())
    ok = mism == 0
    print(f"elements={length}  mismatches={mism}  match={ok}")
    if not ok:
        idx = np.where(src != dst)[0][:8]
        for i in idx:
            print(f"  [{i}] in={src[i]} out={dst[i]}")
        sys.exit(1)


def main():
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 4096

    root = Path(__file__).parent
    build = root / "build" / "hello"
    build.mkdir(parents=True, exist_ok=True)

    mlir = build / "aie.mlir"
    print(f"[1/3] generating MLIR -> {mlir}")
    generate_mlir(length, mlir)

    print(f"[2/3] compiling xclbin in {build}")
    xclbin, insts = compile_xclbin(mlir, build)

    print(f"[3/3] dispatching on NPU (length={length})")
    run_on_device(xclbin, insts, length)


if __name__ == "__main__":
    main()
