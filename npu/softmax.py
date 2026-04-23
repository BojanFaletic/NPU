"""Per-row bf16 softmax on XDNA 2 (AIE2p / Krackan).

Dispatches softmax over the last dim of a bf16 tensor: input [rows, L] -> output
[rows, L] (both bf16). L must be a multiple of 32. Each xclbin is keyed on
(rows, L); rows divides evenly across n_cores (default 1 for MVP).

Built on top of the stock aie_kernels/aie2p/softmax.cc via a thin wrapper
(softmax_kernel.cc) that bakes the row length in at compile time.
"""
from __future__ import annotations
import argparse, subprocess, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from matmul import PEANO, MLIR_AIE, _bf16_to_u16, compile_xclbin


ROOT = Path(__file__).parent
CACHE = ROOT / "build" / "softmax_cache"
WRAPPER_SRC = ROOT / "softmax_kernel.cc"


# -------------------- Peano compile --------------------

def compile_kernel(L: int, build_dir: Path) -> Path:
    """Peano-compile softmax_kernel.cc with SM_LEN=L baked in."""
    obj = build_dir / f"softmax_{L}.o"
    include = MLIR_AIE / "include"
    clang = PEANO / "bin" / "clang++"
    cmd = [
        str(clang),
        "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf", "-DNDEBUG",
        "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
        "-Wno-empty-body", "-Wno-missing-template-arg-list-after-template-kw",
        "-I", str(include),
        f"-DSM_LEN={L}",
        "-c", str(WRAPPER_SRC), "-o", str(obj),
    ]
    print(f"[peano] compiling softmax_{L}.o")
    subprocess.run(cmd, check=True, cwd=build_dir)
    return obj


# -------------------- IRON MLIR generation --------------------

def generate_mlir(rows: int, L: int, n_cores: int, obj_name: str, out_path: Path) -> None:
    """Generate a single-shape softmax IRON program: [rows, L] bf16 -> [rows, L] bf16.

    Each core processes rows/n_cores rows, one row per acquire, via an
    ObjectFifo of shape [L]. Based on vector_exp.py's single-kernel pattern.
    """
    from ml_dtypes import bfloat16 as np_bf16
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2
    from aie.iron.placers import SequentialPlacer

    assert rows % n_cores == 0, f"rows={rows} must be divisible by n_cores={n_cores}"
    assert L % 32 == 0, f"L={L} must be a multiple of SM_VEC_LEN=32"
    tiles_per_core = rows // n_cores

    # Tensor types — flat bf16 buffers
    tensor_ty = np.ndarray[(rows * L,), np.dtype[np_bf16]]
    memtile_ty = np.ndarray[(n_cores * L,), np.dtype[np_bf16]]  # memtile sees one row per core
    tile_ty = np.ndarray[(L,), np.dtype[np_bf16]]

    softmax_k = Kernel("softmax_row_bf16", obj_name, [tile_ty, tile_ty])

    # Data-movement fabric
    A_fifo = ObjectFifo(memtile_ty, name="inA")
    C_fifo = ObjectFifo(memtile_ty, name="outC")
    a_fifos = A_fifo.cons().split(
        offsets=[L * i for i in range(n_cores)], obj_types=[tile_ty] * n_cores
    )
    c_fifos = C_fifo.prod().join(
        offsets=[L * i for i in range(n_cores)], obj_types=[tile_ty] * n_cores
    )

    def core_fn(a_in, c_out, softmax):
        for _ in range_(tiles_per_core) if tiles_per_core > 1 else range(1):
            elem_out = c_out.acquire(1)
            elem_in = a_in.acquire(1)
            softmax(elem_in, elem_out)
            a_in.release(1)
            c_out.release(1)

    workers = []
    for i in range(n_cores):
        workers.append(
            Worker(core_fn, fn_args=[a_fifos[i].cons(), c_fifos[i].prod(), softmax_k])
        )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a_in, c_out):
        rt.start(*workers)
        rt.fill(A_fifo.prod(), a_in)
        rt.drain(C_fifo.cons(), c_out, wait=True)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    out_path.write_text(str(module))


# -------------------- build compiled xclbin --------------------

@dataclass
class Compiled:
    rows: int
    L: int
    n_cores: int
    xclbin_path: Path
    insts: np.ndarray


def build_xclbin(rows: int, L: int, n_cores: int = 1) -> Compiled:
    tag = f"sm_{rows}x{L}_c{n_cores}"
    build = CACHE / tag
    build.mkdir(parents=True, exist_ok=True)
    xclbin = build / "final.xclbin"
    insts  = build / "insts.bin"
    if not xclbin.exists() or not insts.exists():
        obj = compile_kernel(L, build)
        mlir = build / "aie.mlir"
        generate_mlir(rows, L, n_cores, obj.name, mlir)
        compile_xclbin(mlir, obj, build)
    insts_arr = np.fromfile(insts, dtype=np.uint32)
    return Compiled(rows=rows, L=L, n_cores=n_cores, xclbin_path=xclbin, insts=insts_arr)


# -------------------- dispatch via pyxrt --------------------

class _XrtCtx:
    _device = None
    _kernels: dict[Path, tuple] = {}

    @classmethod
    def device(cls):
        if cls._device is None:
            sys.path.insert(0, "/opt/xilinx/xrt/python")
            import pyxrt
            cls._device = pyxrt.device(0)
        return cls._device

    @classmethod
    def kernel_for(cls, c: Compiled):
        if c.xclbin_path in cls._kernels:
            return cls._kernels[c.xclbin_path]
        import pyxrt
        dev = cls.device()
        xb = pyxrt.xclbin(str(c.xclbin_path))
        kname = next(k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE"))
        uuid = dev.register_xclbin(xb)
        ctx = pyxrt.hw_context(dev, uuid)
        kernel = pyxrt.kernel(ctx, kname)
        bo_instr = pyxrt.bo(dev, c.insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
        bo_instr.write(c.insts.tobytes(), 0)
        bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        cls._kernels[c.xclbin_path] = (xb, ctx, kernel, bo_instr)
        return cls._kernels[c.xclbin_path]


class NpuSoftmax:
    """Callable that applies softmax over the last dim of a bf16 tensor.

    Caches one xclbin per (rows, L) shape. Reuses device buffers per shape.
    """
    def __init__(self, n_cores: int = 1):
        self.n_cores = n_cores
        self._compiled: dict[tuple[int, int], Compiled] = {}
        self._bo: dict[tuple[int, int], tuple] = {}

    def _get_compiled(self, rows: int, L: int) -> Compiled:
        key = (rows, L)
        if key not in self._compiled:
            self._compiled[key] = build_xclbin(rows, L, self.n_cores)
        return self._compiled[key]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        import pyxrt
        orig_shape = x.shape
        orig_dtype = x.dtype
        L = orig_shape[-1]
        # The kernel processes vectors in blocks of 32; pad the softmax dim with
        # -inf (which contributes exp(-inf)=0 to the sum, so it's safe to pad).
        L_pad = (L + 31) // 32 * 32
        x2 = x.reshape(-1, L).to(torch.bfloat16)
        rows = x2.shape[0]
        if L_pad != L:
            pad = torch.full((rows, L_pad - L), float("-inf"), dtype=torch.bfloat16)
            x2 = torch.cat([x2, pad], dim=1)

        if rows % self.n_cores != 0:
            raise ValueError(f"rows={rows} not divisible by n_cores={self.n_cores}")

        c = self._get_compiled(rows, L_pad)
        dev = _XrtCtx.device()
        _, _, kernel, bo_instr = _XrtCtx.kernel_for(c)

        nbytes = rows * L_pad * 2  # bf16
        key = (rows, L_pad)
        bo_pair = self._bo.get(key)
        if bo_pair is None:
            bo_a = pyxrt.bo(dev, nbytes, pyxrt.bo.host_only, kernel.group_id(3))
            bo_c = pyxrt.bo(dev, nbytes, pyxrt.bo.host_only, kernel.group_id(4))
            self._bo[key] = (bo_a, bo_c)
        else:
            bo_a, bo_c = bo_pair

        A = x2.contiguous()
        a_np = _bf16_to_u16(A).reshape(-1)
        bo_a.write(a_np.tobytes(), 0)
        bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        run = kernel(3, bo_instr, c.insts.size, bo_a, bo_c)
        run.wait()

        bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out = np.frombuffer(bytes(bo_c.read(nbytes, 0)), dtype=np.uint16).copy()
        out_bf = torch.from_numpy(out).view(torch.bfloat16).reshape(rows, L_pad)[:, :L]

        return out_bf.to(orig_dtype).reshape(orig_shape)


# -------------------- self-test --------------------

def _self_test(rows: int, L: int, n_cores: int) -> None:
    torch.manual_seed(0)
    # Use varied scales per row, so softmax actually needs the max-subtract
    x = torch.randn(rows, L, dtype=torch.bfloat16) * torch.linspace(0.5, 4.0, rows).unsqueeze(1).bfloat16()

    sm = NpuSoftmax(n_cores=n_cores)
    y_npu = sm(x)
    y_fp32 = torch.softmax(x.float(), dim=-1)
    y_bf16 = torch.softmax(x.float(), dim=-1).to(torch.bfloat16).float()

    row_sums = y_npu.float().sum(-1)
    d_fp32 = (y_npu.float() - y_fp32).abs()
    d_bf16 = (y_npu.float() - y_bf16).abs()
    print(f"NpuSoftmax rows={rows} L={L} cores={n_cores}")
    print(f"  vs fp32 softmax: max|Δ|={d_fp32.max().item():.3e}  mean|Δ|={d_fp32.mean().item():.3e}")
    print(f"  vs bf16 softmax: max|Δ|={d_bf16.max().item():.3e}  mean|Δ|={d_bf16.mean().item():.3e}")
    print(f"  row-sum min = {row_sums.min().item():.4f}   max = {row_sums.max().item():.4f}")
    # bf16 softmax has ~3 decimal digits; sum-of-exp divisor is done in bf16,
    # so row sums drift a few permille and per-element error tracks that.
    assert d_fp32.max() < 2e-2, "NPU softmax deviates too much from torch fp32 softmax"
    assert (row_sums - 1.0).abs().max() < 2e-2, "row sums drift from 1"
    print("OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("-L", type=int, default=64)
    ap.add_argument("--cores", type=int, default=1)
    args = ap.parse_args()
    _self_test(args.rows, args.L, args.cores)


if __name__ == "__main__":
    main()
