"""Fused attention on XDNA 2.

Single compute tile runs softmax(Q·K^T / sqrt(D))·V for one (TQ, TK, D) shape
in one dispatch — no CPU roundtrip between the three sub-ops.

Current scope: no causal mask, no multi-block. Correctness-first MVP keyed
against npu/fa_ref.py. Extends to causal + multi-block after this proves out.
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
CACHE = ROOT / "build" / "fa_cache"
KERNEL_SRC = ROOT / "fa_kernel.cc"


def compile_kernel(TQ: int, TK: int, D: int, build_dir: Path) -> Path:
    obj = build_dir / f"fa_{TQ}x{TK}x{D}.o"
    include = MLIR_AIE / "include"
    clang = PEANO / "bin" / "clang++"
    cmd = [
        str(clang),
        "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf", "-DNDEBUG",
        "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
        "-Wno-empty-body", "-Wno-missing-template-arg-list-after-template-kw",
        "-I", str(include),
        f"-DFA_TQ={TQ}", f"-DFA_TK={TK}", f"-DFA_D={D}",
        "-c", str(KERNEL_SRC), "-o", str(obj),
    ]
    print(f"[peano] compiling fa_{TQ}x{TK}x{D}.o")
    subprocess.run(cmd, check=True, cwd=build_dir)
    return obj


def generate_mlir(TQ: int, TK: int, D: int, obj_name: str, out_path: Path) -> None:
    """IRON program: two bf16 buffers (packed QKV -> O), one compute tile.

    Q, K, V are packed into a single input buffer in that order on the host side
    so the compute tile only needs 1 input + 1 output DMA channel (compute
    tiles on AIE2p have a 2-channel input limit).
    """
    from ml_dtypes import bfloat16 as np_bf16
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2
    from aie.iron.placers import SequentialPlacer

    qkv_len = TQ * D + TK * D + TK * D
    qkv_ty = np.ndarray[(qkv_len,), np.dtype[np_bf16]]
    o_ty = np.ndarray[(TQ * D,), np.dtype[np_bf16]]

    attn_k = Kernel("attention_bf16", obj_name, [qkv_ty, o_ty])

    fifo_in = ObjectFifo(qkv_ty, name="inQKV")
    fifo_out = ObjectFifo(o_ty, name="outO")

    def core_fn(of_in, of_out, attn):
        for _ in range_(1):
            elem_out = of_out.acquire(1)
            elem_in = of_in.acquire(1)
            attn(elem_in, elem_out)
            of_in.release(1)
            of_out.release(1)

    # Stack must hold the 2KB S[TQ,TK] scratch + aie_api locals in
    # softmax_simple_bf16 (vector regs are separate, but local accums spill).
    worker = Worker(
        core_fn,
        fn_args=[fifo_in.cons(), fifo_out.prod(), attn_k],
        stack_size=0x2000,
    )

    rt = Runtime()
    with rt.sequence(qkv_ty, o_ty) as (qkv, o):
        rt.start(worker)
        rt.fill(fifo_in.prod(), qkv)
        rt.drain(fifo_out.cons(), o, wait=True)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    out_path.write_text(str(module))


@dataclass
class Compiled:
    TQ: int; TK: int; D: int
    xclbin_path: Path
    insts: np.ndarray


def build_xclbin(TQ: int, TK: int, D: int) -> Compiled:
    tag = f"fa_{TQ}x{TK}x{D}"
    build = CACHE / tag
    build.mkdir(parents=True, exist_ok=True)
    xclbin = build / "final.xclbin"
    insts = build / "insts.bin"
    if not xclbin.exists() or not insts.exists():
        obj = compile_kernel(TQ, TK, D, build)
        mlir = build / "aie.mlir"
        generate_mlir(TQ, TK, D, obj.name, mlir)
        compile_xclbin(mlir, obj, build)
    insts_arr = np.fromfile(insts, dtype=np.uint32)
    return Compiled(TQ=TQ, TK=TK, D=D, xclbin_path=xclbin, insts=insts_arr)


class NpuAttention:
    """Callable: (Q, K, V) -> O, each [*, TQ/TK, D] bf16/fp32, one fused dispatch per head."""
    def __init__(self):
        self._compiled: dict[tuple[int,int,int], Compiled] = {}
        self._bo: dict[tuple[int,int,int], tuple] = {}
        self._kernel_cache: dict[Path, tuple] = {}
        self._device = None

    def _device_obj(self):
        if self._device is None:
            sys.path.insert(0, "/opt/xilinx/xrt/python")
            import pyxrt
            self._device = pyxrt.device(0)
        return self._device

    def _kernel_for(self, c: Compiled):
        import pyxrt
        if c.xclbin_path in self._kernel_cache:
            return self._kernel_cache[c.xclbin_path]
        dev = self._device_obj()
        xb = pyxrt.xclbin(str(c.xclbin_path))
        kname = next(k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE"))
        uuid = dev.register_xclbin(xb)
        ctx = pyxrt.hw_context(dev, uuid)
        kernel = pyxrt.kernel(ctx, kname)
        bo_instr = pyxrt.bo(dev, c.insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
        bo_instr.write(c.insts.tobytes(), 0)
        bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self._kernel_cache[c.xclbin_path] = (xb, ctx, kernel, bo_instr)
        return self._kernel_cache[c.xclbin_path]

    def run_one(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Single-head call. Q [TQ, D], K/V [TK, D]. Returns O [TQ, D] bf16."""
        import pyxrt
        TQ, D = Q.shape
        TK = K.shape[0]
        assert K.shape == (TK, D) and V.shape == (TK, D)

        key = (TQ, TK, D)
        if key not in self._compiled:
            self._compiled[key] = build_xclbin(TQ, TK, D)
        c = self._compiled[key]
        dev = self._device_obj()
        _, _, kernel, bo_instr = self._kernel_for(c)

        qkv_bytes = (TQ * D + TK * D + TK * D) * 2  # bf16
        o_bytes = TQ * D * 2
        if key not in self._bo:
            bo_qkv = pyxrt.bo(dev, qkv_bytes, pyxrt.bo.host_only, kernel.group_id(3))
            bo_o   = pyxrt.bo(dev, o_bytes,   pyxrt.bo.host_only, kernel.group_id(4))
            self._bo[key] = (bo_qkv, bo_o)
        bo_qkv, bo_o = self._bo[key]

        # Pack Q|K|V on host
        QKV = torch.cat([
            Q.to(torch.bfloat16).contiguous().reshape(-1),
            K.to(torch.bfloat16).contiguous().reshape(-1),
            V.to(torch.bfloat16).contiguous().reshape(-1),
        ])
        arr = _bf16_to_u16(QKV).reshape(-1)
        bo_qkv.write(arr.tobytes(), 0)
        bo_qkv.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        run = kernel(3, bo_instr, c.insts.size, bo_qkv, bo_o)
        run.wait()

        bo_o.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out = np.frombuffer(bytes(bo_o.read(o_bytes, 0)), dtype=np.uint16).copy()
        return torch.from_numpy(out).view(torch.bfloat16).reshape(TQ, D)


# -------------------- self-test --------------------

def _torch_attention_nomask(Q, K, V):
    import math
    D = Q.shape[-1]
    S = (Q.float() @ K.float().transpose(-2, -1)) / math.sqrt(D)
    A = torch.softmax(S, dim=-1)
    return A @ V.float()


def _self_test(TQ: int, TK: int, D: int, n_trials: int, fixed_input: bool = False) -> None:
    torch.manual_seed(0)
    attn = NpuAttention()
    fail = []
    if fixed_input:
        Q_f = torch.randn(TQ, D, dtype=torch.bfloat16)
        K_f = torch.randn(TK, D, dtype=torch.bfloat16)
        V_f = torch.randn(TK, D, dtype=torch.bfloat16)
    for trial in range(n_trials):
        if fixed_input:
            Q, K, V = Q_f, K_f, V_f
        else:
            Q = torch.randn(TQ, D, dtype=torch.bfloat16)
            K = torch.randn(TK, D, dtype=torch.bfloat16)
            V = torch.randn(TK, D, dtype=torch.bfloat16)

        O_npu = attn.run_one(Q, K, V).float()
        O_ref = _torch_attention_nomask(Q, K, V)

        nan = torch.isnan(O_npu).any().item()
        inf = torch.isinf(O_npu).any().item()
        d = (O_npu - O_ref).abs()
        max_abs = float("nan") if nan or inf else d.max().item()
        mean_abs = float("nan") if nan or inf else d.mean().item()
        print(f"  trial {trial}: max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  "
              f"{'NaN!' if nan else ''}{'Inf!' if inf else ''}")
        if nan or inf or (max_abs > 5e-2):
            fail.append(trial)

    if fail:
        raise AssertionError(f"{len(fail)}/{n_trials} trials failed: {fail}")
    print(f"NpuAttention TQ={TQ} TK={TK} D={D}: all {n_trials} trials OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--TQ", type=int, default=32)
    ap.add_argument("--TK", type=int, default=32)
    ap.add_argument("-D",  type=int, default=64)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--fixed", action="store_true", help="reuse same Q,K,V every trial")
    args = ap.parse_args()
    _self_test(args.TQ, args.TK, args.D, args.trials, fixed_input=args.fixed)


if __name__ == "__main__":
    main()
