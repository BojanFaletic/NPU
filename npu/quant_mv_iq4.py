"""Decode-time IQ4_XS dequant+matvec on XDNA 2.

This mirrors npu.quant_mv for IQ3_XXS, but targets the routed expert down
weights in Qwen. The on-disk IQ4_XS block is 136 bytes per 256 K values. This
first path expands those values to bf16 once on the host when a matvec object is
created, then the AIE kernel performs native 32-lane bf16 MACs.
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
CACHE = ROOT / "build" / "quant_mv_iq4_cache"
KERNEL_SRC = ROOT / "quant_mv_iq4_kernel.cc"

BLK = 256
RAW_B_BYTES = 136
B_BYTES = 512


def compile_kernel(m: int, build_dir: Path) -> Path:
    obj = build_dir / f"quant_mv_iq4_{m}.o"
    include = MLIR_AIE / "include"
    clang = PEANO / "bin" / "clang++"
    cmd = [
        str(clang),
        "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf", "-DNDEBUG",
        "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
        "-Wno-empty-body", "-Wno-missing-template-arg-list-after-template-kw",
        "-I", str(include),
        f"-DDIM_M={m}",
        "-c", str(KERNEL_SRC), "-o", str(obj),
    ]
    print(f"[peano] compiling quant_mv_iq4_{m}.o")
    subprocess.run(cmd, check=True, cwd=build_dir)
    return obj


def generate_mlir(M: int, K: int, m: int, n_cores: int,
                  obj_name: str, out_path: Path) -> None:
    from ml_dtypes import bfloat16 as np_bf16
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2
    from aie.iron.placers import SequentialPlacer
    from aie.helpers.taplib import TensorTiler2D
    from aie.helpers.taplib.tap import TensorAccessPattern

    assert K % BLK == 0
    assert M % (m * n_cores) == 0

    K_blocks = K // BLK
    M_per_core = M // n_cores
    row_blocks_per_core = M_per_core // m
    row_bytes = K_blocks * B_BYTES

    A_ty = np.ndarray[(M, row_bytes), np.dtype[np.uint8]]
    B_ty = np.ndarray[(1, K), np.dtype[np_bf16]]
    C_ty = np.ndarray[(1, M), np.dtype[np.float32]]
    a_ty = np.ndarray[(m, B_BYTES), np.dtype[np.uint8]]
    b_ty = np.ndarray[(BLK,), np.dtype[np_bf16]]
    c_ty = np.ndarray[(m,), np.dtype[np.float32]]

    zero = Kernel("zero_f32_qmv_iq4", obj_name, [c_ty])
    matvec = Kernel("quant_mv_iq4_xs_bf16_f32", obj_name, [a_ty, b_ty, c_ty])

    fifo_b = ObjectFifo(b_ty, name="inB")
    fifo_a = [ObjectFifo(a_ty, name=f"inA{i}") for i in range(n_cores)]
    fifo_c = [ObjectFifo(c_ty, name=f"outC{i}") for i in range(n_cores)]

    def core_fn(of_a, of_b, of_c, zero_k, matvec_k):
        for _ in range_(row_blocks_per_core) if row_blocks_per_core > 1 else range(1):
            elem_c = of_c.acquire(1)
            zero_k(elem_c)
            for _ in range_(K_blocks) if K_blocks > 1 else range(1):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                matvec_k(elem_a, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    workers = [
        Worker(
            core_fn,
            [fifo_a[i].cons(), fifo_b.cons(), fifo_c[i].prod(), zero, matvec],
            stack_size=0x1000,
        )
        for i in range(n_cores)
    ]

    A_tiles = [
        TensorAccessPattern(
            (M, row_bytes),
            i * M_per_core * row_bytes,
            [row_blocks_per_core, K_blocks, m, B_BYTES],
            [m * row_bytes, B_BYTES, row_bytes, 1],
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
    n_cores: int
    xclbin_path: Path
    insts: np.ndarray


def build_xclbin(M: int, K: int, m: int = 32, n_cores: int = 4) -> Compiled:
    M_pad = ((M + m * n_cores - 1) // (m * n_cores)) * (m * n_cores)
    tag = f"qmv_iq4_{M_pad}x{K}_m{m}_c{n_cores}"
    build = CACHE / tag
    build.mkdir(parents=True, exist_ok=True)
    xclbin = build / "final.xclbin"
    insts = build / "insts.bin"
    stale = (
        not xclbin.exists()
        or not insts.exists()
        or Path(__file__).stat().st_mtime > xclbin.stat().st_mtime
        or KERNEL_SRC.stat().st_mtime > xclbin.stat().st_mtime
        or Path(__file__).stat().st_mtime > insts.stat().st_mtime
        or KERNEL_SRC.stat().st_mtime > insts.stat().st_mtime
    )
    if stale:
        obj = compile_kernel(m, build)
        mlir = build / "aie.mlir"
        generate_mlir(M_pad, K, m, n_cores, obj.name, mlir)
        compile_xclbin(mlir, obj, build)
    return Compiled(M_pad, K, m, n_cores, xclbin,
                    np.fromfile(insts, dtype=np.uint32))


def _repack_iq4_raw(raw: np.ndarray, M: int, K: int) -> np.ndarray:
    from ml_dtypes import bfloat16 as np_bf16

    K_blocks = K // BLK
    raw = np.ascontiguousarray(raw).view(np.uint8).reshape(M, K_blocks, RAW_B_BYTES)

    d = raw[..., :2].view(np.float16).astype(np.float32).reshape(M, K_blocks, 1)
    scales_h = raw[..., 2:4].view(np.uint16).reshape(M, K_blocks, 1)
    scales_l = raw[..., 4:8]
    qs = raw[..., 8:136].reshape(M, K_blocks, 8, 16)

    low_shifts = np.array([0, 4], dtype=np.uint8)
    low = ((scales_l[..., None] >> low_shifts) & np.uint8(0x0F)).reshape(
        M, K_blocks, 8
    )
    high_shifts = np.array([2 * i for i in range(8)], dtype=np.uint16)
    high = ((scales_h >> high_shifts) & np.uint16(0x03)).astype(np.uint8)
    scales = (low | (high << np.uint8(4))).astype(np.int16) - 32
    db = d * scales.astype(np.float32)

    kvalues = np.array(
        [-127, -104, -83, -65, -49, -35, -22, -10,
            1,   13,  25,  38,  53,  69,  89, 113],
        dtype=np.float32,
    )
    low = kvalues[qs & np.uint8(0x0F)]
    high = kvalues[qs >> np.uint8(4)]
    vals = np.concatenate([low, high], axis=-1)
    deq = (db[..., None] * vals).reshape(M, K)
    return np.ascontiguousarray(deq.astype(np_bf16).view(np.uint8))


def repack_iq4_xs_weight(t) -> np.ndarray:
    from gguf import GGMLQuantizationType
    if t.tensor_type != GGMLQuantizationType.IQ4_XS:
        raise ValueError(f"need IQ4_XS tensor, got {t.tensor_type.name}")
    M = int(t.shape[1])
    K = int(t.shape[0])
    K_blocks = K // BLK
    raw = np.ascontiguousarray(t.data).view(np.uint8).reshape(M, K_blocks * RAW_B_BYTES)
    return _repack_iq4_raw(raw, M, K)


def repack_iq4_xs_per_expert(t, expert_idx: int) -> np.ndarray:
    from gguf import GGMLQuantizationType
    if t.tensor_type != GGMLQuantizationType.IQ4_XS:
        raise ValueError(f"need IQ4_XS tensor, got {t.tensor_type.name}")
    if len(t.shape) != 3:
        raise ValueError(f"need 3D stacked expert tensor, got {tuple(t.shape)}")
    n_expert = int(t.shape[-1])
    if not 0 <= expert_idx < n_expert:
        raise IndexError(expert_idx)
    K = int(t.shape[0])
    M = int(t.shape[1])
    K_blocks = K // BLK
    raw = np.ascontiguousarray(t.data[expert_idx]).view(np.uint8)
    raw = raw.reshape(M, K_blocks * RAW_B_BYTES)
    return _repack_iq4_raw(raw, M, K)


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
        kname = next(k.get_name() for k in xb.get_kernels()
                     if k.get_name().startswith("MLIR_AIE"))
        uuid = dev.register_xclbin(xb)
        ctx = pyxrt.hw_context(dev, uuid)
        kernel = pyxrt.kernel(ctx, kname)
        bo_instr = pyxrt.bo(dev, c.insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
        bo_instr.write(c.insts.tobytes(), 0)
        bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        cls._kernels[c.xclbin_path] = (xb, ctx, kernel, bo_instr)
        return cls._kernels[c.xclbin_path]


class NpuIQ4MatVec:
    """Single-vector F.linear(x, W) for IQ4_XS weights expanded to bf16."""

    def __init__(self, packed_weight: np.ndarray, M: int, K: int,
                 n_cores: int = 4, m: int | None = None):
        if packed_weight.dtype != np.uint8:
            raise TypeError(f"packed_weight must be uint8, got {packed_weight.dtype}")
        if K % BLK != 0:
            raise ValueError(f"K={K} must be a multiple of {BLK}")
        K_blocks = K // BLK
        expected_bytes = M * K_blocks * B_BYTES
        if packed_weight.size != expected_bytes:
            raise ValueError(
                f"packed_weight has {packed_weight.size} bytes, expected "
                f"{expected_bytes} for M={M} K={K}"
            )
        self.out_features = M
        self.in_features = K
        self.n_cores = n_cores
        if m is None:
            m = 32
        self._compiled = build_xclbin(M, K, m=m, n_cores=n_cores)

        row_bytes = K_blocks * B_BYTES
        packed = np.ascontiguousarray(packed_weight).reshape(M, row_bytes)
        if self._compiled.M != M:
            pad_rows = self._compiled.M - M
            zero_pad = np.zeros((pad_rows, row_bytes), dtype=np.uint8)
            packed = np.concatenate([packed, zero_pad], axis=0)
        self._W: np.ndarray | None = np.ascontiguousarray(packed)

        self._bo_w: pyxrt.bo | None = None
        self._bo_x: pyxrt.bo | None = None
        self._bo_y: pyxrt.bo | None = None

    @classmethod
    def from_iq4_xs_bytes(cls, packed: np.ndarray, M: int, K: int,
                          n_cores: int = 4,
                          m: int | None = None) -> "NpuIQ4MatVec":
        return cls(packed, M, K, n_cores=n_cores, m=m)

    @classmethod
    def from_gguf_tensor(cls, t, expert_idx: int | None = None,
                         n_cores: int = 4,
                         m: int | None = None) -> "NpuIQ4MatVec":
        if expert_idx is None:
            packed = repack_iq4_xs_weight(t)
            M = int(t.shape[1])
            K = int(t.shape[0])
        else:
            packed = repack_iq4_xs_per_expert(t, expert_idx)
            M = int(t.shape[1])
            K = int(t.shape[0])
        return cls(packed, M, K, n_cores=n_cores, m=m)

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
            if self._W is None:
                raise RuntimeError("NPU weight staging buffer was already released")
            self._bo_w = pyxrt.bo(dev, self._W.nbytes, pyxrt.bo.host_only,
                                  kernel.group_id(3))
            self._bo_w.write(self._W.tobytes(), 0)
            self._bo_w.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._W = None

        if self._bo_x is None:
            self._bo_x = pyxrt.bo(dev, self.in_features * 2,
                                  pyxrt.bo.host_only, kernel.group_id(4))
            self._bo_y = pyxrt.bo(dev, c.M * 4, pyxrt.bo.host_only,
                                  kernel.group_id(5))

        with profile("qmv_iq4.bf16"):
            x_bf = x1.to(torch.bfloat16).contiguous()
            x_np = _bf16_to_u16(x_bf).reshape(-1)
        with profile("qmv_iq4.write"):
            self._bo_x.write(x_np.tobytes(), 0)
        with profile("qmv_iq4.sync_to"):
            self._bo_x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        with profile("qmv_iq4.kernel"):
            run = kernel(3, bo_instr, c.insts.size,
                         self._bo_w, self._bo_x, self._bo_y)
            run.wait()
        with profile("qmv_iq4.sync_from"):
            self._bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        with profile("qmv_iq4.read"):
            y = np.frombuffer(bytes(self._bo_y.read(c.M * 4, 0)), dtype=np.float32)
            y = y[:self.out_features].copy()
        with profile("qmv_iq4.to_torch"):
            return torch.from_numpy(y).reshape(*x.shape[:-1], self.out_features)


def _self_test() -> None:
    from ml_dtypes import bfloat16 as np_bf16
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import IQ4_XS as GGUF_IQ4
    path = Path(__file__).resolve().parent.parent / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
    r = GGUFReader(str(path))
    t = next(t for t in r.tensors
             if t.tensor_type == GGMLQuantizationType.IQ4_XS and len(t.shape) == 3)
    e = 17
    packed = repack_iq4_xs_per_expert(t, e)
    K = int(t.shape[0])
    M = int(t.shape[1])
    print(f"using {t.name} expert {e}  M={M} K={K} packed={packed.shape}")

    raw = np.ascontiguousarray(t.data[e]).view(np.uint8)
    ref = GGUF_IQ4.dequantize_blocks(raw.reshape(-1, RAW_B_BYTES)).reshape(M, K)
    ours = packed.view(np_bf16).astype(np.float32).reshape(M, K)
    diff = np.abs(ours - ref).max()
    print(f"repack bf16 max|d|={diff:.3e}")
    assert diff < 1e-2


if __name__ == "__main__":
    _self_test()
