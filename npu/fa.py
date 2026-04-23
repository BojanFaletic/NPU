"""FlashAttention-2 forward on XDNA 2.

One dispatch processes one Q block (BR rows) attending to all TK keys. The
host streams n_kv = TK/BC key/value block pairs through the compute tile;
the kernel keeps running softmax state (m, l, O_acc) in static tile memory
across the block calls so the [BR, TK] intermediate never materialises.

MVP scope: no causal mask yet, single Q block per dispatch (if Tq > BR,
caller loops). Correctness oracle is npu/fa_ref.py.
"""
from __future__ import annotations
import argparse, math, subprocess, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from matmul import PEANO, MLIR_AIE, _bf16_to_u16, compile_xclbin


ROOT = Path(__file__).parent
CACHE = ROOT / "build" / "fa_cache"
KERNEL_SRC = ROOT / "fa_kernel.cc"


def compile_kernel(BR: int, BC: int, D: int, build_dir: Path) -> Path:
    obj = build_dir / f"fa_{BR}x{BC}x{D}.o"
    include = MLIR_AIE / "include"
    clang = PEANO / "bin" / "clang++"
    cmd = [
        str(clang),
        "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf", "-DNDEBUG",
        "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
        "-Wno-empty-body", "-Wno-missing-template-arg-list-after-template-kw",
        "-I", str(include),
        f"-DFA_BR={BR}", f"-DFA_BC={BC}", f"-DFA_D={D}",
        "-c", str(KERNEL_SRC), "-o", str(obj),
    ]
    print(f"[peano] compiling fa_{BR}x{BC}x{D}.o")
    subprocess.run(cmd, check=True, cwd=build_dir)
    return obj


HEADER_BF16 = 32  # matches fa_kernel.cc; header riding in Q buffer


def generate_mlir(BR: int, BC: int, D: int, n_kv: int, obj_name: str, out_path: Path) -> None:
    """Two input FIFOs (Q+header once + KV-pairs streamed n_kv times) -> O.

    Compute tiles have a 2-input DMA channel limit, so Q carries start_row and
    the causal flag inline in its first 32 bf16 lanes, and K/V are paired into
    one streaming FIFO (each tile is [K | V] concatenated, 2·BC·D bf16).
    """
    from ml_dtypes import bfloat16 as np_bf16
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2
    from aie.iron.placers import SequentialPlacer

    q_len  = HEADER_BF16 + BR * D
    kv_len = 2 * BC * D
    o_len  = BR * D

    q_ty  = np.ndarray[(q_len,),  np.dtype[np_bf16]]
    kv_ty = np.ndarray[(kv_len,), np.dtype[np_bf16]]
    o_ty  = np.ndarray[(o_len,),  np.dtype[np_bf16]]
    kv_stream_ty = np.ndarray[(n_kv * kv_len,), np.dtype[np_bf16]]

    block_k    = Kernel("attn_block",    obj_name, [q_ty, kv_ty])
    finalise_k = Kernel("attn_finalise", obj_name, [o_ty])

    fifo_q  = ObjectFifo(q_ty,  name="inQ")
    fifo_kv = ObjectFifo(kv_ty, name="inKV")
    fifo_o  = ObjectFifo(o_ty,  name="outO")

    def core_fn(of_q, of_kv, of_o, attn_block, attn_finalise):
        elem_o = of_o.acquire(1)
        elem_q = of_q.acquire(1)
        for _ in range_(n_kv) if n_kv > 1 else range(1):
            elem_kv = of_kv.acquire(1)
            attn_block(elem_q, elem_kv)
            of_kv.release(1)
        attn_finalise(elem_o)
        of_q.release(1)
        of_o.release(1)

    worker = Worker(
        core_fn,
        fn_args=[fifo_q.cons(), fifo_kv.cons(), fifo_o.prod(), block_k, finalise_k],
        # g_O[BR*DH] fp32 = 8KB static + S[BR*BC] fp32 = 4KB stack + call frames
        stack_size=0x3000,
    )

    rt = Runtime()
    with rt.sequence(q_ty, kv_stream_ty, o_ty) as (q, kv, o):
        rt.start(worker)
        rt.fill(fifo_q.prod(),  q)
        rt.fill(fifo_kv.prod(), kv)
        rt.drain(fifo_o.cons(), o, wait=True)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    out_path.write_text(str(module))


@dataclass
class Compiled:
    BR: int; BC: int; D: int; n_kv: int
    xclbin_path: Path
    insts: np.ndarray


def build_xclbin(BR: int, BC: int, D: int, n_kv: int) -> Compiled:
    tag = f"fa_{BR}x{BC}x{D}_n{n_kv}"
    build = CACHE / tag
    build.mkdir(parents=True, exist_ok=True)
    xclbin = build / "final.xclbin"
    insts  = build / "insts.bin"
    if not xclbin.exists() or not insts.exists():
        obj = compile_kernel(BR, BC, D, build)
        mlir = build / "aie.mlir"
        generate_mlir(BR, BC, D, n_kv, obj.name, mlir)
        compile_xclbin(mlir, obj, build)
    insts_arr = np.fromfile(insts, dtype=np.uint32)
    return Compiled(BR=BR, BC=BC, D=D, n_kv=n_kv, xclbin_path=xclbin, insts=insts_arr)


class NpuAttention:
    """(Q, K, V) -> O for one head, one Q block.

    Q: [BR, D], K/V: [n_kv*BC, D]. D and BR hard-coded against the xclbin
    shape (recompile for new dims). n_kv is inferred from TK/BC.
    """
    def __init__(self, BR: int = 32, BC: int = 32, D: int = 64):
        self.BR, self.BC, self.D = BR, BC, D
        self._compiled: dict[int, Compiled] = {}   # keyed by n_kv
        self._bo: dict[int, tuple] = {}
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

    def run_one(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        start_row: int = 0, causal: bool = False,
    ) -> torch.Tensor:
        """Q [BR, D], K [TK, D], V [TK, D], TK multiple of BC. Returns O [BR, D] bf16.

        start_row is the absolute position of Q row 0 in the full sequence; it
        drives the causal mask alongside the block index. Used when Tq > BR and
        this call is handling a middle Q block (rows start_row..start_row+BR-1).
        """
        import pyxrt
        BR, D = Q.shape
        TK = K.shape[0]
        BC = self.BC
        assert (BR, D) == (self.BR, self.D), f"Q shape {Q.shape} != ({self.BR}, {self.D})"
        assert K.shape == (TK, D) and V.shape == (TK, D)
        assert TK % BC == 0, f"TK={TK} not a multiple of BC={BC}"
        n_kv = TK // BC

        if n_kv not in self._compiled:
            self._compiled[n_kv] = build_xclbin(BR, BC, D, n_kv)
        c = self._compiled[n_kv]
        dev = self._device_obj()
        _, _, kernel, bo_instr = self._kernel_for(c)

        q_bytes  = (HEADER_BF16 + BR * D) * 2  # header + Q data, bf16
        kv_bytes = n_kv * 2 * BC * D * 2
        o_bytes  = BR * D * 2
        if n_kv not in self._bo:
            bo_q  = pyxrt.bo(dev, q_bytes,  pyxrt.bo.host_only, kernel.group_id(3))
            bo_kv = pyxrt.bo(dev, kv_bytes, pyxrt.bo.host_only, kernel.group_id(4))
            bo_o  = pyxrt.bo(dev, o_bytes,  pyxrt.bo.host_only, kernel.group_id(5))
            self._bo[n_kv] = (bo_q, bo_kv, bo_o)
        bo_q, bo_kv, bo_o = self._bo[n_kv]

        # Q buffer = header (start_row, causal, padding) + Q data
        header = torch.zeros(HEADER_BF16, dtype=torch.bfloat16)
        header[0] = float(start_row)
        header[1] = 1.0 if causal else 0.0
        q_payload = torch.cat([header, Q.to(torch.bfloat16).contiguous().reshape(-1)])
        q_arr = _bf16_to_u16(q_payload).reshape(-1)
        bo_q.write(q_arr.tobytes(), 0)
        bo_q.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        K_bf = K.to(torch.bfloat16).contiguous()
        V_bf = V.to(torch.bfloat16).contiguous()
        kv_blocks = []
        for j in range(n_kv):
            kv_blocks.append(K_bf[j*BC:(j+1)*BC].reshape(-1))
            kv_blocks.append(V_bf[j*BC:(j+1)*BC].reshape(-1))
        kv = torch.cat(kv_blocks)
        kv_arr = _bf16_to_u16(kv).reshape(-1)
        bo_kv.write(kv_arr.tobytes(), 0)
        bo_kv.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        run = kernel(3, bo_instr, c.insts.size, bo_q, bo_kv, bo_o)
        run.wait()

        bo_o.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out = np.frombuffer(bytes(bo_o.read(o_bytes, 0)), dtype=np.uint16).copy()
        return torch.from_numpy(out).view(torch.bfloat16).reshape(BR, D)


# -------------------- self-test --------------------

def _torch_attention_nomask(Q, K, V):
    D = Q.shape[-1]
    S = (Q.float() @ K.float().transpose(-2, -1)) / math.sqrt(D)
    A = torch.softmax(S, dim=-1)
    return A @ V.float()


def _torch_causal_attention(Q, K, V, start_row: int):
    """Standard-attention reference (what smollm.py computes on CPU path).

    Q rows map to absolute positions [start_row, start_row+BR); K cols to [0, TK).
    Causal mask: col c attends to row r iff c <= start_row + r.
    """
    Tq = Q.shape[0]
    Tk = K.shape[0]
    D = Q.shape[-1]
    S = (Q.float() @ K.float().transpose(-2, -1)) / math.sqrt(D)
    row = torch.arange(Tq)[:, None] + start_row
    col = torch.arange(Tk)[None, :]
    mask = torch.where(col <= row, 0.0, float("-inf")).to(S.dtype)
    S = S + mask
    A = torch.softmax(S, dim=-1)
    return A @ V.float()


def _self_test(BR: int, BC: int, D: int, TK: int, n_trials: int, causal: bool) -> None:
    torch.manual_seed(0)
    attn = NpuAttention(BR=BR, BC=BC, D=D)
    fail = []
    for trial in range(n_trials):
        Q = torch.randn(BR, D, dtype=torch.bfloat16)
        K = torch.randn(TK, D, dtype=torch.bfloat16)
        V = torch.randn(TK, D, dtype=torch.bfloat16)

        # For causal, vary start_row per trial so we exercise partial masks.
        start_row = (trial * BR) % max(TK, BR) if causal else 0

        O_npu = attn.run_one(Q, K, V, start_row=start_row, causal=causal).float()
        if causal:
            O_ref = _torch_causal_attention(Q, K, V, start_row=start_row)
        else:
            O_ref = _torch_attention_nomask(Q, K, V)

        nan = torch.isnan(O_npu).any().item()
        inf = torch.isinf(O_npu).any().item()
        d = (O_npu - O_ref).abs()
        max_abs = float("nan") if nan or inf else d.max().item()
        mean_abs = float("nan") if nan or inf else d.mean().item()
        flag = ("NaN!" if nan else "") + ("Inf!" if inf else "")
        tag = f"causal sr={start_row}" if causal else "nomask"
        print(f"  trial {trial} [{tag}]: max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  {flag}")
        if nan or inf or (max_abs > 1e-1):
            fail.append(trial)

    if fail:
        raise AssertionError(f"{len(fail)}/{n_trials} trials failed: {fail}")
    print(f"NpuAttention BR={BR} BC={BC} D={D} TK={TK} (n_kv={TK//BC}) "
          f"causal={causal}: all {n_trials} trials OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--BR", type=int, default=32)
    ap.add_argument("--BC", type=int, default=32)
    ap.add_argument("-D",  type=int, default=64)
    ap.add_argument("--TK", type=int, default=64)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()
    _self_test(args.BR, args.BC, args.D, args.TK, args.trials, args.causal)


if __name__ == "__main__":
    main()
