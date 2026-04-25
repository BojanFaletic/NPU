"""Decode-time bf16 matrix-vector linear on XDNA 2.

This is a prototype for T=1 projection/MLP decode. It computes
`F.linear(x, W)` for a single vector without routing through the padded
matrix-matrix path (`M=1 -> M_pad=256`). We start with a scalar AIE kernel to
measure whether the dataflow and launch shape are worth vectorizing/fusing.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from matmul import PEANO, MLIR_AIE, compile_xclbin, _bf16_to_u16

sys.path.insert(0, "/opt/xilinx/xrt/python")
import pyxrt


ROOT = Path(__file__).parent
CACHE = ROOT / "build" / "mv_cache"
KERNEL_SRC = ROOT / "mv_kernel.cc"


def compile_kernel(m: int, k: int, build_dir: Path) -> Path:
    obj = build_dir / f"mv_{m}x{k}.o"
    include = MLIR_AIE / "include"
    clang = PEANO / "bin" / "clang++"
    cmd = [
        str(clang),
        "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf", "-DNDEBUG",
        "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
        "-Wno-empty-body", "-Wno-missing-template-arg-list-after-template-kw",
        "-I", str(include),
        f"-DDIM_M={m}", f"-DDIM_K={k}",
        "-c", str(KERNEL_SRC), "-o", str(obj),
    ]
    print(f"[peano] compiling mv_{m}x{k}.o")
    subprocess.run(cmd, check=True, cwd=build_dir)
    return obj


def generate_mlir(M: int, K: int, m: int, k: int, n_cores: int,
                  obj_name: str, out_path: Path) -> None:
    from ml_dtypes import bfloat16 as np_bf16
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2
    from aie.iron.placers import SequentialPlacer
    from aie.helpers.taplib import TensorTiler2D
    from aie.helpers.taplib.tap import TensorAccessPattern

    assert M % (m * n_cores) == 0
    assert K % k == 0

    M_per_core = M // n_cores
    row_blocks_per_core = M_per_core // m
    K_div_k = K // k

    A_ty = np.ndarray[(M, K), np.dtype[np_bf16]]
    B_ty = np.ndarray[(1, K), np.dtype[np_bf16]]
    C_ty = np.ndarray[(1, M), np.dtype[np.float32]]
    a_ty = np.ndarray[(m, k), np.dtype[np_bf16]]
    b_ty = np.ndarray[(k,), np.dtype[np_bf16]]
    c_ty = np.ndarray[(m,), np.dtype[np.float32]]

    zero = Kernel("zero_f32", obj_name, [c_ty])
    matvec = Kernel("matvec_bf16_f32", obj_name, [a_ty, b_ty, c_ty])

    fifo_b = ObjectFifo(b_ty, name="inB")
    fifo_a = [ObjectFifo(a_ty, name=f"inA{i}") for i in range(n_cores)]
    fifo_c = [ObjectFifo(c_ty, name=f"outC{i}") for i in range(n_cores)]

    def core_fn(of_a, of_b, of_c, zero_k, matvec_k):
        for _ in range_(row_blocks_per_core) if row_blocks_per_core > 1 else range(1):
            elem_c = of_c.acquire(1)
            zero_k(elem_c)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                matvec_k(elem_a, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    workers = [
        Worker(core_fn, [fifo_a[i].cons(), fifo_b.cons(), fifo_c[i].prod(), zero, matvec])
        for i in range(n_cores)
    ]

    # Each core owns a contiguous slice of output rows. A is tiled by
    # [row-block, k-block, m, k] within each slice.
    A_tiles = [
        TensorAccessPattern(
            (M, K),
            i * M_per_core * K,
            [row_blocks_per_core, K_div_k, m, k],
            [m * K, k, K, 1],
        )
        for i in range(n_cores)
    ]

    C_tiles = TensorTiler2D.simple_tiler((1, M), (1, M_per_core), prune_step=False)
    b_tap = TensorTiler2D.simple_tiler(
        (1, K), pattern_repeat=row_blocks_per_core, prune_step=False,
    )[0]

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)
        rt.fill(fifo_b.prod(), B, tap=b_tap)
        for i in range(n_cores):
            rt.fill(fifo_a[i].prod(), A, tap=A_tiles[i])
            rt.drain(fifo_c[i].cons(), C, tap=C_tiles[i], wait=True)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    out_path.write_text(str(module))


@dataclass
class Compiled:
    M: int
    K: int
    m: int
    k: int
    n_cores: int
    xclbin_path: Path
    insts: np.ndarray


def build_xclbin(M: int, K: int, m: int = 32, k: int = 64,
                 n_cores: int = 8) -> Compiled:
    M_pad = ((M + m * n_cores - 1) // (m * n_cores)) * (m * n_cores)
    tag = f"mv_{M_pad}x{K}_{m}x{k}_c{n_cores}"
    build = CACHE / tag
    build.mkdir(parents=True, exist_ok=True)
    xclbin = build / "final.xclbin"
    insts = build / "insts.bin"
    stale = (
        not xclbin.exists()
        or not insts.exists()
        or KERNEL_SRC.stat().st_mtime > xclbin.stat().st_mtime
        or KERNEL_SRC.stat().st_mtime > insts.stat().st_mtime
    )
    if stale:
        obj = compile_kernel(m, k, build)
        mlir = build / "aie.mlir"
        generate_mlir(M_pad, K, m, k, n_cores, obj.name, mlir)
        compile_xclbin(mlir, obj, build)
    return Compiled(M_pad, K, m, k, n_cores, xclbin, np.fromfile(insts, dtype=np.uint32))


class _XrtCtx:
    _device: pyxrt.device | None = None
    _kernels: dict[Path, tuple[pyxrt.xclbin, pyxrt.hw_context, pyxrt.kernel, pyxrt.bo]] = {}

    @classmethod
    def device(cls) -> pyxrt.device:
        if cls._device is None:
            cls._device = pyxrt.device(0)
        return cls._device

    @classmethod
    def kernel_for(cls, c: Compiled):
        if c.xclbin_path in cls._kernels:
            return cls._kernels[c.xclbin_path]
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


class NpuMatVec:
    """Single-vector F.linear(x, W) using a matrix-vector NPU program."""

    def __init__(self, weight: torch.Tensor, n_cores: int = 8):
        assert weight.dim() == 2
        self.out_features, self.in_features = weight.shape
        self.n_cores = n_cores
        self._compiled = build_xclbin(self.out_features, self.in_features, n_cores=n_cores)
        self._weight_src: torch.Tensor | None = weight
        self._W: torch.Tensor | None = None
        self._bo_w: pyxrt.bo | None = None
        self._bo_x: pyxrt.bo | None = None
        self._bo_y: pyxrt.bo | None = None

    def _stage_weight(self) -> torch.Tensor:
        if self._weight_src is None:
            raise RuntimeError("NPU source weight was already released")
        W = self._weight_src.to(torch.bfloat16).contiguous()
        if self._compiled.M != self.out_features:
            W = torch.cat([
                W,
                torch.zeros(self._compiled.M - self.out_features, self.in_features,
                            dtype=torch.bfloat16),
            ], dim=0).contiguous()
        return W

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from npu.profiler import profile
        except ModuleNotFoundError:
            from profiler import profile

        x1 = x.reshape(-1)
        assert x1.numel() == self.in_features
        c = self._compiled
        dev = _XrtCtx.device()
        _, _, kernel, bo_instr = _XrtCtx.kernel_for(c)

        if self._bo_w is None:
            W = self._stage_weight()
            w_np = _bf16_to_u16(W).reshape(-1)
            self._bo_w = pyxrt.bo(dev, w_np.nbytes, pyxrt.bo.host_only, kernel.group_id(3))
            self._bo_w.write(w_np.tobytes(), 0)
            self._bo_w.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._weight_src = None
            self._W = None

        if self._bo_x is None:
            self._bo_x = pyxrt.bo(dev, self.in_features * 2, pyxrt.bo.host_only, kernel.group_id(4))
            self._bo_y = pyxrt.bo(dev, c.M * 4, pyxrt.bo.host_only, kernel.group_id(5))

        with profile("mv.bf16"):
            x_bf = x1.to(torch.bfloat16).contiguous()
            x_np = _bf16_to_u16(x_bf).reshape(-1)
        with profile("mv.write"):
            self._bo_x.write(x_np.tobytes(), 0)
        with profile("mv.sync_to"):
            self._bo_x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        with profile("mv.kernel"):
            run = kernel(3, bo_instr, c.insts.size, self._bo_w, self._bo_x, self._bo_y)
            run.wait()
        with profile("mv.sync_from"):
            self._bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        with profile("mv.read"):
            y = np.frombuffer(bytes(self._bo_y.read(c.M * 4, 0)), dtype=np.float32)
            y = y[:self.out_features].copy()
        with profile("mv.to_torch"):
            return torch.from_numpy(y).reshape(*x.shape[:-1], self.out_features)


def _self_test():
    torch.manual_seed(0)
    W = torch.randn(576, 576, dtype=torch.bfloat16)
    x = torch.randn(576, dtype=torch.float32)
    mv = NpuMatVec(W)
    y = mv(x)
    ref = torch.nn.functional.linear(x.to(torch.bfloat16).float(), W.float())
    diff = (y - ref).abs()
    print(f"NpuMatVec max|Δ|={diff.max().item():.3e} mean|Δ|={diff.mean().item():.3e}")
    assert diff.max().item() < 5e-2


if __name__ == "__main__":
    _self_test()
