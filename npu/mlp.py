"""Decode-time fused MLP on XDNA 2.

Computes the SmolLM decode MLP for one token in one NPU dispatch:

    y = down(silu(gate(x)) * up(x))

The implementation keeps the two weight matrices resident in BOs, uses f32 for
the gate/up intermediate, stores the activation as bf16, then runs the down
matvec. This is intentionally fixed-shape for the 135M model dimensions first;
if it measures well we can generalize it.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from matmul import PEANO, MLIR_AIE, compile_xclbin, _bf16_to_u16

sys.path.insert(0, "/opt/xilinx/xrt/python")
import pyxrt


ROOT = Path(__file__).parent
CACHE = ROOT / "build" / "mlp_cache"
KERNEL_SRC = ROOT / "mlp_kernel.cc"


def compile_kernel(m: int, k: int, act: int, build_dir: Path) -> Path:
    obj = build_dir / f"mlp_{m}x{k}_act{act}.o"
    include = MLIR_AIE / "include"
    clang = PEANO / "bin" / "clang++"
    cmd = [
        str(clang),
        "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf", "-DNDEBUG",
        "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
        "-Wno-empty-body", "-Wno-missing-template-arg-list-after-template-kw",
        "-I", str(include),
        f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_ACT={act}",
        "-c", str(KERNEL_SRC), "-o", str(obj),
    ]
    print(f"[peano] compiling {obj.name}")
    subprocess.run(cmd, check=True, cwd=build_dir)
    return obj


def _vec_tap(total: int, offset: int, blocks: int, block: int):
    from aie.helpers.taplib.tap import TensorAccessPattern

    return TensorAccessPattern(
        (1, total),
        offset,
        [blocks, 1, 1, block],
        [block, 0, 0, 1],
    )


def _pack_weight_for_split(
    weight: torch.Tensor,
    M: int,
    K: int,
    m: int,
    k: int,
    n_cores: int,
) -> torch.Tensor:
    """Pack row-split weights as [row_block, k_block, core, m, k]."""
    M_per_core = M // n_cores
    return (
        weight.reshape(n_cores, M_per_core, K)
        .reshape(n_cores, M_per_core // m, m, K // k, k)
        .permute(1, 3, 0, 2, 4)
        .contiguous()
        .reshape(-1)
    )


def generate_mlir(
    hidden: int,
    ffn: int,
    down_m: int,
    m: int,
    k: int,
    act: int,
    n_cores: int,
    obj_name: str,
    out_path: Path,
) -> None:
    from ml_dtypes import bfloat16 as np_bf16
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2
    from aie.iron.placers import SequentialPlacer
    from aie.helpers.taplib import TensorTiler2D
    from aie.helpers.taplib.tap import TensorAccessPattern

    gu_m = 2 * ffn
    assert hidden % k == 0
    assert ffn % k == 0
    assert gu_m % (m * n_cores) == 0
    assert down_m % (m * n_cores) == 0
    assert ffn % (act * n_cores) == 0

    gu_m_per_core = gu_m // n_cores
    gu_row_blocks = gu_m_per_core // m
    gu_k_div = hidden // k

    act_per_core = ffn // n_cores
    act_blocks = act_per_core // act

    down_m_per_core = down_m // n_cores
    down_row_blocks = down_m_per_core // m
    down_k_div = ffn // k

    Wgu_ty = np.ndarray[(gu_m * hidden,), np.dtype[np_bf16]]
    Wdown_ty = np.ndarray[(down_m * ffn,), np.dtype[np_bf16]]
    XA_ty = np.ndarray[(1, hidden + ffn), np.dtype[np_bf16]]
    Tmp_ty = np.ndarray[(1, gu_m), np.dtype[np.float32]]
    Y_ty = np.ndarray[(1, down_m), np.dtype[np.float32]]

    a_gu_ty = np.ndarray[(m, k), np.dtype[np_bf16]]
    b_gu_ty = np.ndarray[(k,), np.dtype[np_bf16]]
    c_gu_ty = np.ndarray[(m,), np.dtype[np.float32]]
    a_gu_mem_ty = np.ndarray[(n_cores * m * k,), np.dtype[np_bf16]]
    a_down_ty = np.ndarray[(m, k), np.dtype[np_bf16]]
    b_down_ty = np.ndarray[(k,), np.dtype[np_bf16]]
    c_down_ty = np.ndarray[(m,), np.dtype[np.float32]]
    a_down_mem_ty = np.ndarray[(n_cores * m * k,), np.dtype[np_bf16]]
    act_in_ty = np.ndarray[(act,), np.dtype[np.float32]]
    act_out_ty = np.ndarray[(act,), np.dtype[np_bf16]]

    zero = Kernel("zero_f32", obj_name, [c_gu_ty])
    matvec = Kernel("matvec_bf16_f32", obj_name, [a_gu_ty, b_gu_ty, c_gu_ty])
    swiglu = Kernel("swiglu_f32_bf16", obj_name, [act_in_ty, act_in_ty, act_out_ty])

    gu_b = ObjectFifo(b_gu_ty, name="guB")
    gu_a_mem = ObjectFifo(a_gu_mem_ty, name="guA")
    gu_a = gu_a_mem.cons().split(
        offsets=[m * k * i for i in range(n_cores)],
        obj_types=[a_gu_ty] * n_cores,
    )
    gu_c = [ObjectFifo(c_gu_ty, name=f"guC{i}") for i in range(n_cores)]

    act_gate = [ObjectFifo(act_in_ty, name=f"actGate{i}") for i in range(n_cores)]
    act_up = [ObjectFifo(act_in_ty, name=f"actUp{i}") for i in range(n_cores)]
    act_out = [ObjectFifo(act_out_ty, name=f"actOut{i}") for i in range(n_cores)]

    down_b = ObjectFifo(b_down_ty, name="downB")
    down_a_mem = ObjectFifo(a_down_mem_ty, name="downA")
    down_a = down_a_mem.cons().split(
        offsets=[m * k * i for i in range(n_cores)],
        obj_types=[a_down_ty] * n_cores,
    )
    down_c = [ObjectFifo(c_down_ty, name=f"downC{i}") for i in range(n_cores)]

    def gate_up_core_fn(of_gu_a, of_gu_b, of_gu_c, zero_k, matvec_k):
        for _ in range_(gu_row_blocks) if gu_row_blocks > 1 else range(1):
            elem_c = of_gu_c.acquire(1)
            zero_k(elem_c)
            for _ in range_(gu_k_div) if gu_k_div > 1 else range(1):
                elem_a = of_gu_a.acquire(1)
                elem_b = of_gu_b.acquire(1)
                matvec_k(elem_a, elem_b, elem_c)
                of_gu_a.release(1)
                of_gu_b.release(1)
            of_gu_c.release(1)

    def act_core_fn(of_act_gate, of_act_up, of_act_out, swiglu_k):
        for _ in range_(act_blocks) if act_blocks > 1 else range(1):
            elem_gate = of_act_gate.acquire(1)
            elem_up = of_act_up.acquire(1)
            elem_out = of_act_out.acquire(1)
            swiglu_k(elem_gate, elem_up, elem_out)
            of_act_gate.release(1)
            of_act_up.release(1)
            of_act_out.release(1)

    def down_core_fn(of_down_a, of_down_b, of_down_c, zero_k, matvec_k):
        for _ in range_(down_row_blocks) if down_row_blocks > 1 else range(1):
            elem_c = of_down_c.acquire(1)
            zero_k(elem_c)
            for _ in range_(down_k_div) if down_k_div > 1 else range(1):
                elem_a = of_down_a.acquire(1)
                elem_b = of_down_b.acquire(1)
                matvec_k(elem_a, elem_b, elem_c)
                of_down_a.release(1)
                of_down_b.release(1)
            of_down_c.release(1)

    gu_workers = [
        Worker(gate_up_core_fn, [gu_a[i].cons(), gu_b.cons(), gu_c[i].prod(), zero, matvec])
        for i in range(n_cores)
    ]
    act_workers = [
        Worker(act_core_fn, [act_gate[i].cons(), act_up[i].cons(), act_out[i].prod(), swiglu])
        for i in range(n_cores)
    ]
    down_workers = [
        Worker(down_core_fn, [down_a[i].cons(), down_b.cons(), down_c[i].prod(), zero, matvec])
        for i in range(n_cores)
    ]

    gu_c_tiles = TensorTiler2D.simple_tiler((1, gu_m), (1, gu_m_per_core), prune_step=False)
    gu_b_tap = TensorAccessPattern(
        (1, hidden + ffn),
        0,
        [gu_row_blocks, 1, 1, hidden],
        [0, 0, 0, 1],
    )

    act_gate_taps = [
        _vec_tap(gu_m, i * act_per_core, act_blocks, act)
        for i in range(n_cores)
    ]
    act_up_taps = [
        _vec_tap(gu_m, ffn + i * act_per_core, act_blocks, act)
        for i in range(n_cores)
    ]
    act_out_taps = [
        _vec_tap(hidden + ffn, hidden + i * act_per_core, act_blocks, act)
        for i in range(n_cores)
    ]

    down_c_tiles = TensorTiler2D.simple_tiler((1, down_m), (1, down_m_per_core), prune_step=False)
    down_b_tap = TensorAccessPattern(
        (1, hidden + ffn),
        hidden,
        [down_row_blocks, 1, 1, ffn],
        [0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(Wgu_ty, Wdown_ty, XA_ty, Tmp_ty, Y_ty) as (Wgu, Wdown, XA, Tmp, Y):
        rt.start(*gu_workers, *act_workers, *down_workers)

        tg = rt.task_group()
        rt.fill(gu_b.prod(), XA, tap=gu_b_tap, task_group=tg)
        rt.fill(gu_a_mem.prod(), Wgu, task_group=tg)
        for i in range(n_cores):
            rt.drain(gu_c[i].cons(), Tmp, tap=gu_c_tiles[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)

        tg = rt.task_group()
        for i in range(n_cores):
            rt.fill(act_gate[i].prod(), Tmp, tap=act_gate_taps[i], task_group=tg)
            rt.fill(act_up[i].prod(), Tmp, tap=act_up_taps[i], task_group=tg)
            rt.drain(act_out[i].cons(), XA, tap=act_out_taps[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)

        tg = rt.task_group()
        rt.fill(down_b.prod(), XA, tap=down_b_tap, task_group=tg)
        rt.fill(down_a_mem.prod(), Wdown, task_group=tg)
        for i in range(n_cores):
            rt.drain(down_c[i].cons(), Y, tap=down_c_tiles[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    out_path.write_text(str(module))


@dataclass
class Compiled:
    hidden: int
    ffn: int
    down_m: int
    m: int
    k: int
    act: int
    n_cores: int
    xclbin_path: Path
    insts: np.ndarray


def build_xclbin(
    hidden: int = 576,
    ffn: int = 1536,
    m: int = 32,
    k: int = 64,
    act: int = 32,
    n_cores: int = 4,
) -> Compiled:
    down_m = ((hidden + m * n_cores - 1) // (m * n_cores)) * (m * n_cores)
    tag = f"mlp_h{hidden}_f{ffn}_down{down_m}_{m}x{k}_a{act}_c{n_cores}"
    build = CACHE / tag
    build.mkdir(parents=True, exist_ok=True)
    xclbin = build / "final.xclbin"
    insts = build / "insts.bin"
    stale = (
        not xclbin.exists()
        or not insts.exists()
        or KERNEL_SRC.stat().st_mtime > xclbin.stat().st_mtime
        or KERNEL_SRC.stat().st_mtime > insts.stat().st_mtime
        or Path(__file__).stat().st_mtime > xclbin.stat().st_mtime
        or Path(__file__).stat().st_mtime > insts.stat().st_mtime
    )
    if stale:
        obj = compile_kernel(m, k, act, build)
        mlir = build / "aie.mlir"
        generate_mlir(hidden, ffn, down_m, m, k, act, n_cores, obj.name, mlir)
        compile_xclbin(mlir, obj, build)
    return Compiled(hidden, ffn, down_m, m, k, act, n_cores, xclbin, np.fromfile(insts, dtype=np.uint32))


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


class NpuFusedMLP:
    """Single-token fused MLP using gate/up, activation, and down on the NPU."""

    def __init__(self, w_gate_up: torch.Tensor, w_down: torch.Tensor, n_cores: int = 4):
        assert w_gate_up.dim() == 2 and w_down.dim() == 2
        self.ffn2, self.hidden = w_gate_up.shape
        self.hidden_out, self.ffn = w_down.shape
        assert self.ffn2 == 2 * self.ffn
        assert self.hidden_out == self.hidden
        self.n_cores = n_cores
        self._compiled = build_xclbin(self.hidden, self.ffn, n_cores=n_cores)
        Wgu = w_gate_up.to(torch.bfloat16).contiguous()
        Wdown = w_down.to(torch.bfloat16).contiguous()
        if self._compiled.down_m != self.hidden:
            Wdown = torch.cat([
                Wdown,
                torch.zeros(
                    self._compiled.down_m - self.hidden,
                    self.ffn,
                    dtype=torch.bfloat16,
                ),
            ], dim=0).contiguous()
        self._Wgu: torch.Tensor | None = _pack_weight_for_split(
            Wgu, self.ffn2, self.hidden, self._compiled.m, self._compiled.k, self.n_cores,
        )
        self._Wdown: torch.Tensor | None = _pack_weight_for_split(
            Wdown, self._compiled.down_m, self.ffn, self._compiled.m, self._compiled.k, self.n_cores,
        )
        self._bo_wgu: pyxrt.bo | None = None
        self._bo_wdown: pyxrt.bo | None = None
        self._bo_xa: pyxrt.bo | None = None
        self._bo_tmp: pyxrt.bo | None = None
        self._bo_y: pyxrt.bo | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from npu.profiler import profile
        except ModuleNotFoundError:
            from profiler import profile

        x1 = x.reshape(-1)
        assert x1.numel() == self.hidden
        c = self._compiled
        dev = _XrtCtx.device()
        _, _, kernel, bo_instr = _XrtCtx.kernel_for(c)

        if self._bo_wgu is None:
            if self._Wgu is None or self._Wdown is None:
                raise RuntimeError("NPU weight staging buffers were already released")
            wgu_np = _bf16_to_u16(self._Wgu).reshape(-1)
            wdown_np = _bf16_to_u16(self._Wdown).reshape(-1)
            self._bo_wgu = pyxrt.bo(dev, wgu_np.nbytes, pyxrt.bo.host_only, kernel.group_id(3))
            self._bo_wdown = pyxrt.bo(dev, wdown_np.nbytes, pyxrt.bo.host_only, kernel.group_id(4))
            self._bo_wgu.write(wgu_np.tobytes(), 0)
            self._bo_wdown.write(wdown_np.tobytes(), 0)
            self._bo_wgu.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._bo_wdown.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._Wgu = None
            self._Wdown = None

        if self._bo_xa is None:
            self._bo_xa = pyxrt.bo(dev, (self.hidden + self.ffn) * 2, pyxrt.bo.host_only, kernel.group_id(5))
            self._bo_tmp = pyxrt.bo(dev, self.ffn2 * 4, pyxrt.bo.host_only, kernel.group_id(6))
            self._bo_y = pyxrt.bo(dev, c.down_m * 4, pyxrt.bo.host_only, kernel.group_id(7))

        with profile("mlp.bf16"):
            x_bf = x1.to(torch.bfloat16).contiguous()
            x_np = _bf16_to_u16(x_bf).reshape(-1)
        with profile("mlp.write"):
            self._bo_xa.write(x_np.tobytes(), 0)
        with profile("mlp.sync_to"):
            self._bo_xa.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        with profile("mlp.kernel"):
            run = kernel(
                3,
                bo_instr,
                c.insts.size,
                self._bo_wgu,
                self._bo_wdown,
                self._bo_xa,
                self._bo_tmp,
                self._bo_y,
            )
            run.wait()
        with profile("mlp.sync_from"):
            self._bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        with profile("mlp.read"):
            y = np.frombuffer(bytes(self._bo_y.read(c.down_m * 4, 0)), dtype=np.float32)
            y = y[:self.hidden].copy()
        with profile("mlp.to_torch"):
            return torch.from_numpy(y).reshape(*x.shape[:-1], self.hidden)


def _self_test():
    torch.manual_seed(0)
    hidden, ffn = 576, 1536
    w_gate = torch.randn(ffn, hidden, dtype=torch.float32) * 0.02
    w_up = torch.randn(ffn, hidden, dtype=torch.float32) * 0.02
    w_down = torch.randn(hidden, ffn, dtype=torch.float32) * 0.02
    w_gate_up = torch.cat([w_gate, w_up], dim=0).to(torch.bfloat16)
    w_down = w_down.to(torch.bfloat16)
    x = torch.randn(hidden, dtype=torch.float32)
    mlp = NpuFusedMLP(w_gate_up, w_down)
    y = mlp(x)

    x_bf = x.to(torch.bfloat16).float()
    gu = F.linear(x_bf, w_gate_up.float())
    act = (F.silu(gu[:ffn]) * gu[ffn:]).to(torch.bfloat16)
    ref = F.linear(act.float(), w_down.float())
    diff = (y - ref).abs()
    print(f"NpuFusedMLP max|Δ|={diff.max().item():.3e} mean|Δ|={diff.mean().item():.3e}")
    assert diff.max().item() < 1e-1


if __name__ == "__main__":
    _self_test()
