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


def generate_mlir(BR: int, BC: int, D: int, n_kv: int, n_q_total: int,
                  obj_name: str, out_path: Path) -> None:
    """Two input FIFOs (Q+header stream, KV-pair stream) -> O stream. One
    dispatch processes n_q_total Q-blocks (amortises host-side dispatch
    overhead across all heads × Q-blocks in a layer).

    Q carries start_row + causal flag inline; compute tiles have a 2-input
    DMA channel limit so K and V are paired into one streaming FIFO.
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
    q_stream_ty  = np.ndarray[(n_q_total * q_len,),         np.dtype[np_bf16]]
    kv_stream_ty = np.ndarray[(n_q_total * n_kv * kv_len,), np.dtype[np_bf16]]
    o_stream_ty  = np.ndarray[(n_q_total * o_len,),         np.dtype[np_bf16]]

    block_k    = Kernel("attn_block",    obj_name, [q_ty, kv_ty])
    finalise_k = Kernel("attn_finalise", obj_name, [o_ty])

    fifo_q  = ObjectFifo(q_ty,  name="inQ")
    fifo_kv = ObjectFifo(kv_ty, name="inKV")
    fifo_o  = ObjectFifo(o_ty,  name="outO")

    def core_fn(of_q, of_kv, of_o, attn_block, attn_finalise):
        for _ in range_(n_q_total) if n_q_total > 1 else range(1):
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
        # g_O[BR*DH] bf16 = 4KB + g_m/g_l = 256B + S[BR*BC] fp32 = 4KB stack
        stack_size=0x3000,
    )

    rt = Runtime()
    with rt.sequence(q_stream_ty, kv_stream_ty, o_stream_ty) as (q, kv, o):
        rt.start(worker)
        rt.fill(fifo_q.prod(),  q)
        rt.fill(fifo_kv.prod(), kv)
        rt.drain(fifo_o.cons(), o, wait=True)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    out_path.write_text(str(module))


@dataclass
class Compiled:
    BR: int; BC: int; D: int; n_kv: int; n_q_total: int
    xclbin_path: Path
    insts: np.ndarray


def build_xclbin(BR: int, BC: int, D: int, n_kv: int, n_q_total: int) -> Compiled:
    tag = f"fa_{BR}x{BC}x{D}_n{n_kv}_q{n_q_total}"
    build = CACHE / tag
    build.mkdir(parents=True, exist_ok=True)
    xclbin = build / "final.xclbin"
    insts  = build / "insts.bin"
    if not xclbin.exists() or not insts.exists():
        obj = compile_kernel(BR, BC, D, build)
        mlir = build / "aie.mlir"
        generate_mlir(BR, BC, D, n_kv, n_q_total, obj.name, mlir)
        compile_xclbin(mlir, obj, build)
    insts_arr = np.fromfile(insts, dtype=np.uint32)
    return Compiled(BR=BR, BC=BC, D=D, n_kv=n_kv, n_q_total=n_q_total,
                    xclbin_path=xclbin, insts=insts_arr)


class NpuAttention:
    """Callable that runs FA on a batch of Q-blocks in one NPU dispatch.

    Batching across Q-blocks amortises dispatch overhead (PCIe upload +
    Python marshalling + kernel invoke). xclbins are keyed on (n_kv, n_q)
    so a full prefill reuses the same xclbin across layers.
    """
    def __init__(self, BR: int = 32, BC: int = 32, D: int = 64):
        self.BR, self.BC, self.D = BR, BC, D
        self._compiled: dict[tuple[int, int], Compiled] = {}   # (n_kv, n_q_total)
        self._bo: dict[tuple[int, int], tuple] = {}
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

    def run_batch(
        self,
        Q_stack: torch.Tensor,     # [N, BR, D]
        K_stack: torch.Tensor,     # [N, TK, D] — per-Q-block K (dup across Q-blocks of same head)
        V_stack: torch.Tensor,     # [N, TK, D]
        start_rows: list[int],     # length N
        causal: bool = True,
    ) -> torch.Tensor:
        """Return O_stack [N, BR, D] bf16. TK must be a multiple of BC."""
        import pyxrt
        N, BR, D = Q_stack.shape
        TK = K_stack.shape[1]
        BC = self.BC
        assert (BR, D) == (self.BR, self.D)
        assert K_stack.shape == (N, TK, D) and V_stack.shape == (N, TK, D)
        assert TK % BC == 0
        assert len(start_rows) == N
        n_kv = TK // BC

        key = (n_kv, N)
        if key not in self._compiled:
            self._compiled[key] = build_xclbin(BR, BC, D, n_kv, N)
        c = self._compiled[key]
        dev = self._device_obj()
        _, _, kernel, bo_instr = self._kernel_for(c)

        q_tile_len  = HEADER_BF16 + BR * D
        kv_tile_len = 2 * BC * D
        o_tile_len  = BR * D
        q_bytes  = N * q_tile_len * 2
        kv_bytes = N * n_kv * kv_tile_len * 2
        o_bytes  = N * o_tile_len * 2
        if key not in self._bo:
            bo_q  = pyxrt.bo(dev, q_bytes,  pyxrt.bo.host_only, kernel.group_id(3))
            bo_kv = pyxrt.bo(dev, kv_bytes, pyxrt.bo.host_only, kernel.group_id(4))
            bo_o  = pyxrt.bo(dev, o_bytes,  pyxrt.bo.host_only, kernel.group_id(5))
            self._bo[key] = (bo_q, bo_kv, bo_o)
        bo_q, bo_kv, bo_o = self._bo[key]

        # Native bf16 mac shape for aie::mmul on AIE2p
        R_MAC, S_MAC, T_MAC = 4, 8, 8
        MR, MK, MT = BR // R_MAC, D // S_MAC, BC // T_MAC

        # Q: [N, BR, D] -> tiled [N, MR, MK, R_MAC, S_MAC] contiguous so each
        # 4×8 A-tile is one vector load in the kernel.
        Q_bf = Q_stack.to(torch.bfloat16).contiguous()
        Q_tiled = Q_bf.view(N, MR, R_MAC, MK, S_MAC).permute(0, 1, 3, 2, 4).contiguous()

        # Header (start_row, causal) then Q_tiled.
        headers = torch.zeros((N, HEADER_BF16), dtype=torch.bfloat16)
        headers[:, 0] = torch.tensor(start_rows, dtype=torch.bfloat16)
        headers[:, 1] = 1.0 if causal else 0.0
        q_tiles = torch.cat([headers, Q_tiled.reshape(N, -1)], dim=1)
        q_arr = _bf16_to_u16(q_tiles.reshape(-1)).reshape(-1)
        bo_q.write(q_arr.tobytes(), 0)
        bo_q.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # K: [N, TK, D] = [N, n_kv, BC, D]. For each kv-block, tile K as the B
        # operand of mmul for S = Q·K^T, i.e. K_tiled[k_chunk, j_chunk, s, t]
        # = K[j_chunk*t + t, k_chunk*s + s].
        K_bf = K_stack.to(torch.bfloat16).contiguous().view(N, n_kv, MT, T_MAC, MK, S_MAC)
        K_tiled = K_bf.permute(0, 1, 4, 2, 5, 3).contiguous()  # [N, n_kv, MK, MT, S, T]

        # V stays row-major [N, n_kv, BC, D] — the S·V path uses scalar-P_row
        # vector MAC over the BC dim; tiling V for mmul is correct but its
        # win is eaten by the scatter-P / gather-O work around the matmul.
        V_bf = V_stack.to(torch.bfloat16).contiguous().view(N, n_kv, BC, D)

        kv_blocks = []
        for ni in range(N):
            for j in range(n_kv):
                kv_blocks.append(K_tiled[ni, j].reshape(-1))
                kv_blocks.append(V_bf[ni, j].reshape(-1))
        kv = torch.cat(kv_blocks)
        kv_arr = _bf16_to_u16(kv).reshape(-1)
        bo_kv.write(kv_arr.tobytes(), 0)
        bo_kv.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        run = kernel(3, bo_instr, c.insts.size, bo_q, bo_kv, bo_o)
        run.wait()

        bo_o.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out = np.frombuffer(bytes(bo_o.read(o_bytes, 0)), dtype=np.uint16).copy()
        return torch.from_numpy(out).view(torch.bfloat16).reshape(N, BR, D)

    def run_one(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        start_row: int = 0, causal: bool = False,
    ) -> torch.Tensor:
        """Single-Q-block wrapper around run_batch (kept for the self-test)."""
        O = self.run_batch(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0),
                           [start_row], causal=causal)
        return O.squeeze(0)


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
