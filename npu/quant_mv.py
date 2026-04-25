"""Decode-time IQ3_XXS dequant+matvec on XDNA 2.

Wraps a quantized weight matrix as a callable equivalent to F.linear(x, W),
where the weight is stored on-device as IQ3_XXS bytes and dequantized inside
the AIE tile (4 KB GRID + 1 KB KSIGNS_FP LUTs in tile DM).

Per-block layout (host-preprocessed): 100 bytes
  [ 0: 4]   d (fp32; the on-disk fp16 d is widened to fp32 by the host)
  [ 4:68]   qs[64]   uint8 grid indices into the 256-entry × 4-lane GRID
  [68:100]  scales_signs[8]  uint32 — bits[31:28] = sub-block scale,
                                       bits[27:0] = 4 × 7-bit sign indices

Per-row total bytes: K_blocks * 100, where K_blocks = K / 256.

Constraints (today):
  - K must be a multiple of 256 (K_blocks > 0).
  - M is padded to a multiple of `m * n_cores`. The public wrapper picks
    m=128 for Qwen-sized expert rows and m=32 for smaller matrices unless
    overridden.
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
from iq3_xxs import GRID, KSIGNS, dequant_rows

sys.path.insert(0, "/opt/xilinx/xrt/python")
import pyxrt


ROOT = Path(__file__).parent
CACHE = ROOT / "build" / "quant_mv_cache"
KERNEL_SRC = ROOT / "quant_mv_kernel.cc"
TABLES_HDR = ROOT / "iq3_xxs_tables.h"

BLK = 256          # K elements per IQ3_XXS block
B_BYTES = 100      # bytes per block (host-preprocessed: fp32 d, 64 qs, 32 scales)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _emit_tables_header() -> None:
    """Generate iq3_xxs_tables.h from the Python LUTs. Re-run if stale."""
    ksigns_fp = KSIGNS.astype(np.float32)
    lines: list[str] = [
        "// Auto-generated from npu/iq3_xxs.py — do not hand-edit.",
        "#pragma once",
        "#include <stdint.h>",
        "",
        "alignas(64) static const float IQ3_XXS_GRID[256 * 4] = {",
    ]
    for r in GRID:
        lines.append("  " + ", ".join(f"{v:.9f}f" for v in r) + ",")
    lines.append("};")
    lines.append("alignas(64) static const float IQ3_XXS_KSIGNS_FP[128 * 8] = {")
    for r in ksigns_fp:
        lines.append("  " + ", ".join(f"{v:+.1f}f" for v in r) + ",")
    lines.append("};")
    text = "\n".join(lines) + "\n"
    if not TABLES_HDR.exists() or TABLES_HDR.read_text() != text:
        TABLES_HDR.write_text(text)


def compile_kernel(m: int, build_dir: Path) -> Path:
    obj = build_dir / f"quant_mv_{m}.o"
    include = MLIR_AIE / "include"
    clang = PEANO / "bin" / "clang++"
    _emit_tables_header()
    cmd = [
        str(clang),
        "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf", "-DNDEBUG",
        "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
        "-Wno-empty-body", "-Wno-missing-template-arg-list-after-template-kw",
        "-I", str(include),
        "-I", str(ROOT),
        f"-DDIM_M={m}",
        "-c", str(KERNEL_SRC), "-o", str(obj),
    ]
    print(f"[peano] compiling quant_mv_{m}.o")
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

    # Per-row weight bytes: K_blocks * B_BYTES.  Whole-W byte dimension: M rows.
    row_bytes = K_blocks * B_BYTES

    A_ty = np.ndarray[(M, row_bytes), np.dtype[np.uint8]]
    B_ty = np.ndarray[(1, K), np.dtype[np_bf16]]
    C_ty = np.ndarray[(1, M), np.dtype[np.float32]]
    a_ty = np.ndarray[(m, B_BYTES), np.dtype[np.uint8]]
    b_ty = np.ndarray[(BLK,), np.dtype[np_bf16]]
    c_ty = np.ndarray[(m,), np.dtype[np.float32]]

    zero = Kernel("zero_f32_qmv", obj_name, [c_ty])
    matvec = Kernel("quant_mv_iq3_xxs_bf16_f32", obj_name, [a_ty, b_ty, c_ty])

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

    # Each core owns a contiguous slice of output rows. Within each slice, A is
    # tiled by [row-block, k-block, m, B_BYTES]: offset(row_block, k_block, im)
    # = (row_block * m + im) * row_bytes + k_block * B_BYTES.
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


def build_xclbin(M: int, K: int, m: int = 128, n_cores: int = 4) -> Compiled:
    M_pad = ((M + m * n_cores - 1) // (m * n_cores)) * (m * n_cores)
    tag = f"qmv_{M_pad}x{K}_m{m}_c{n_cores}"
    build = CACHE / tag
    build.mkdir(parents=True, exist_ok=True)
    xclbin = build / "final.xclbin"
    insts = build / "insts.bin"
    src_mtime = max(Path(__file__).stat().st_mtime,
                    (ROOT / "iq3_xxs.py").stat().st_mtime,
                    KERNEL_SRC.stat().st_mtime,
                    TABLES_HDR.stat().st_mtime if TABLES_HDR.exists() else 0)
    stale = (
        not xclbin.exists()
        or not insts.exists()
        or src_mtime > xclbin.stat().st_mtime
        or src_mtime > insts.stat().st_mtime
    )
    if stale:
        obj = compile_kernel(m, build)
        mlir = build / "aie.mlir"
        generate_mlir(M_pad, K, m, n_cores, obj.name, mlir)
        compile_xclbin(mlir, obj, build)
    return Compiled(M_pad, K, m, n_cores, xclbin,
                    np.fromfile(insts, dtype=np.uint32))


# ---------------------------------------------------------------------------
# Host-side weight repacking: GGUF tensor → (M, row_bytes) uint8.
# ---------------------------------------------------------------------------

def repack_iq3_xxs_weight(t) -> np.ndarray:
    """Repack a GGUF IQ3_XXS tensor (shape (in, out) on disk; per-row 98-byte
    blocks) into the kernel-friendly (M=out, K_blocks * 100) uint8 buffer.

    Notes:
      - Per-block d is widened from fp16 to fp32 (2 → 4 bytes) so the kernel
        doesn't have to materialize an fp16 cast.
      - The input weight tensor is GGML-row-major: the fast axis is `in` and
        each row's blocks are contiguous.
    """
    from gguf import GGMLQuantizationType
    if t.tensor_type != GGMLQuantizationType.IQ3_XXS:
        raise ValueError(f"need IQ3_XXS tensor, got {t.tensor_type.name}")
    raw = np.ascontiguousarray(t.data).view(np.uint8)

    # Disk layout for a 2D tensor: (n_rows_per_disk_axis_1, bytes_per_row).
    # GGML shape is (in, out), but on disk t.data is reshaped (out, bytes_per_row_in_K)
    # by gguf-py's loader. We treat raw as (M, row_bytes_98) where M is the
    # *output* dim (rows of the unfolded matrix).
    M = int(t.shape[1])
    in_dim = int(t.shape[0])
    K_blocks = in_dim // BLK
    row_bytes_98 = K_blocks * 98
    raw = raw.reshape(M, row_bytes_98)

    # Per-block widen d (fp16 → fp32). New row_bytes = K_blocks * 100.
    out = np.empty((M, K_blocks, B_BYTES), dtype=np.uint8)
    blk_in = raw.reshape(M, K_blocks, 98)
    # d (fp16, 2 bytes) → fp32 (4 bytes)
    d_fp16 = blk_in[..., :2].view(np.float16)            # (M, K_blocks, 1)
    d_fp32 = d_fp16.astype(np.float32)                   # (M, K_blocks, 1)
    out[..., :4] = d_fp32.view(np.uint8).reshape(M, K_blocks, 4)
    out[..., 4:68] = blk_in[..., 2:66]                   # qs
    out[..., 68:100] = blk_in[..., 66:98]                # scales_signs
    return out.reshape(M, K_blocks * B_BYTES)


def repack_iq3_xxs_per_expert(t, expert_idx: int) -> np.ndarray:
    """Same as repack_iq3_xxs_weight but for one expert of a 3D stacked tensor.

    Stacked tensor disk layout: t.data has shape (n_expert, rows_per_expert,
    bytes_per_row). Each expert's bytes are contiguous because the per-expert
    element count (in*out) is a multiple of the IQ3_XXS block size."""
    from gguf import GGMLQuantizationType
    if t.tensor_type != GGMLQuantizationType.IQ3_XXS:
        raise ValueError(f"need IQ3_XXS tensor, got {t.tensor_type.name}")
    if len(t.shape) != 3:
        raise ValueError(f"need 3D stacked expert tensor, got {tuple(t.shape)}")
    n_expert = int(t.shape[-1])
    if not 0 <= expert_idx < n_expert:
        raise IndexError(expert_idx)

    in_dim = int(t.shape[0])
    M = int(t.shape[1])
    K_blocks = in_dim // BLK
    row_bytes_98 = K_blocks * 98

    # gguf-py stores stacked data as t.data shape (n_expert, M_per_expert, bytes/row)
    raw = np.ascontiguousarray(t.data[expert_idx]).view(np.uint8)
    raw = raw.reshape(M, row_bytes_98)
    blk_in = raw.reshape(M, K_blocks, 98)

    out = np.empty((M, K_blocks, B_BYTES), dtype=np.uint8)
    d_fp16 = blk_in[..., :2].view(np.float16)
    d_fp32 = d_fp16.astype(np.float32)
    out[..., :4] = d_fp32.view(np.uint8).reshape(M, K_blocks, 4)
    out[..., 4:68] = blk_in[..., 2:66]
    out[..., 68:100] = blk_in[..., 66:98]
    return out.reshape(M, K_blocks * B_BYTES)


# ---------------------------------------------------------------------------
# XRT context (shared with mv.py would be cleaner; duplicate-on-purpose to keep
# the modules independent for now).
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NpuQuantMatVec
# ---------------------------------------------------------------------------

class NpuQuantMatVec:
    """Single-vector F.linear(x, W) where W is stored as IQ3_XXS bytes
    on-device. Equivalent to F.linear(x, dequant(W)).

    Two ways to construct:
      - NpuQuantMatVec.from_iq3_xxs_bytes(packed_u8, M, K) : already
        host-repacked weight (output of repack_iq3_xxs_*).
      - NpuQuantMatVec.from_gguf_tensor(t [, expert_idx]) : pulls and
        repacks from a GGUF tensor.
    """

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
            m = 128 if M >= 512 else 32
        self._compiled = build_xclbin(M, K, m=m, n_cores=n_cores)

        # Pad rows to M_pad if needed.
        row_bytes = K_blocks * B_BYTES
        packed = np.ascontiguousarray(packed_weight).reshape(M, row_bytes)
        if self._compiled.M != M:
            pad_rows = self._compiled.M - M
            zero_pad = np.zeros((pad_rows, row_bytes), dtype=np.uint8)
            packed = np.concatenate([packed, zero_pad], axis=0)
        self._W = np.ascontiguousarray(packed)

        self._bo_w: pyxrt.bo | None = None
        self._bo_x: pyxrt.bo | None = None
        self._bo_y: pyxrt.bo | None = None

    @classmethod
    def from_iq3_xxs_bytes(cls, packed: np.ndarray, M: int, K: int,
                           n_cores: int = 4,
                           m: int | None = None) -> "NpuQuantMatVec":
        return cls(packed, M, K, n_cores=n_cores, m=m)

    @classmethod
    def from_gguf_tensor(cls, t, expert_idx: int | None = None,
                         n_cores: int = 4,
                         m: int | None = None) -> "NpuQuantMatVec":
        if expert_idx is None:
            packed = repack_iq3_xxs_weight(t)
            M = int(t.shape[1])
            K = int(t.shape[0])
        else:
            packed = repack_iq3_xxs_per_expert(t, expert_idx)
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
            self._bo_w = pyxrt.bo(dev, self._W.nbytes, pyxrt.bo.host_only,
                                  kernel.group_id(3))
            self._bo_w.write(self._W.tobytes(), 0)
            self._bo_w.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        if self._bo_x is None:
            self._bo_x = pyxrt.bo(dev, self.in_features * 2,
                                  pyxrt.bo.host_only, kernel.group_id(4))
            self._bo_y = pyxrt.bo(dev, c.M * 4, pyxrt.bo.host_only,
                                  kernel.group_id(5))

        with profile("qmv.bf16"):
            x_bf = x1.to(torch.bfloat16).contiguous()
            x_np = _bf16_to_u16(x_bf).reshape(-1)
        with profile("qmv.write"):
            self._bo_x.write(x_np.tobytes(), 0)
        with profile("qmv.sync_to"):
            self._bo_x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        with profile("qmv.kernel"):
            run = kernel(3, bo_instr, c.insts.size,
                         self._bo_w, self._bo_x, self._bo_y)
            run.wait()
        with profile("qmv.sync_from"):
            self._bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        with profile("qmv.read"):
            y = np.frombuffer(bytes(self._bo_y.read(c.M * 4, 0)), dtype=np.float32)
            y = y[:self.out_features].copy()
        with profile("qmv.to_torch"):
            return torch.from_numpy(y).reshape(*x.shape[:-1], self.out_features)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Repack a real expert IQ3_XXS tensor and verify host-side dequant is
    bit-equal to gguf-py. (NPU run is exercised by tests/test_npu_experts.py.)"""
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import IQ3_XXS as GGUF_IQ3
    path = Path(__file__).resolve().parent.parent / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
    r = GGUFReader(str(path))
    t = next(t for t in r.tensors
             if t.tensor_type == GGMLQuantizationType.IQ3_XXS)
    print(f"using {t.name}  shape={tuple(int(s) for s in t.shape)}")

    if len(t.shape) == 3:
        packed = repack_iq3_xxs_per_expert(t, 17)
        in_dim = int(t.shape[0])
        M = int(t.shape[1])
        # gguf-py dequant of one expert
        chunk = np.ascontiguousarray(t.data[17]).view(np.uint8)
        dr_full = GGUF_IQ3.dequantize_blocks(chunk.reshape(-1, 98)).reshape(-1)
    else:
        packed = repack_iq3_xxs_weight(t)
        in_dim = int(t.shape[0])
        M = int(t.shape[1])
        chunk = np.ascontiguousarray(t.data).view(np.uint8)
        dr_full = GGUF_IQ3.dequantize_blocks(chunk.reshape(-1, 98)).reshape(-1)

    K_blocks = in_dim // BLK
    expected_bytes = M * K_blocks * B_BYTES
    print(f"  M={M} K={in_dim} K_blocks={K_blocks}  packed bytes={packed.size} (expected {expected_bytes})")
    assert packed.size == expected_bytes

    # Re-dequant from our packed format and compare against gguf-py.
    # Repack back to original 98-byte form for dequant_rows compatibility.
    packed_3d = packed.reshape(M, K_blocks, B_BYTES)
    rebuilt_98 = np.empty((M, K_blocks, 98), dtype=np.uint8)
    # Convert d back fp32 → fp16 (lossless for the original fp16 values)
    d_fp32 = packed_3d[..., :4].view(np.float32).reshape(M, K_blocks, 1)
    d_fp16 = d_fp32.astype(np.float16)
    rebuilt_98[..., :2] = d_fp16.view(np.uint8).reshape(M, K_blocks, 2)
    rebuilt_98[..., 2:66] = packed_3d[..., 4:68]
    rebuilt_98[..., 66:98] = packed_3d[..., 68:100]
    dr_ours = dequant_rows(rebuilt_98).reshape(-1)
    diff = np.abs(dr_ours - dr_full).max()
    print(f"  repack round-trip max|Δ|={diff:.3e}")
    assert diff == 0.0


if __name__ == "__main__":
    _self_test()
