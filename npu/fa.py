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
                  n_cores: int, obj_name: str, out_path: Path) -> None:
    """n_q_total Q-blocks processed in one dispatch, split across n_cores
    compute tiles via memtile-aggregated ObjectFifo.split/join.

    Memtile layout: each FIFO aggregates n_cores tiles per memtile iteration;
    split() chops by per-core offset so core i gets tile i of each memtile.
    Total iterations per core = n_q_total / n_cores (outer Q loop) and
    n_q_total / n_cores * n_kv (KV stream inner). Host lays data out as
    [iter_0 core_0, iter_0 core_1, ..., iter_0 core_{n_cores-1}, iter_1 core_0, ...].
    """
    from ml_dtypes import bfloat16 as np_bf16
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2
    from aie.iron.placers import SequentialPlacer

    assert n_q_total % n_cores == 0, f"n_q_total={n_q_total} not divisible by n_cores={n_cores}"
    n_per_core = n_q_total // n_cores

    q_len  = HEADER_BF16 + BR * D
    kv_len = 2 * BC * D
    o_len  = BR * D

    q_tile_ty  = np.ndarray[(q_len,),  np.dtype[np_bf16]]
    kv_tile_ty = np.ndarray[(kv_len,), np.dtype[np_bf16]]
    o_tile_ty  = np.ndarray[(o_len,),  np.dtype[np_bf16]]

    q_mem_ty  = np.ndarray[(n_cores * q_len,),  np.dtype[np_bf16]]
    kv_mem_ty = np.ndarray[(n_cores * kv_len,), np.dtype[np_bf16]]
    o_mem_ty  = np.ndarray[(n_cores * o_len,),  np.dtype[np_bf16]]

    q_stream_ty  = np.ndarray[(n_q_total * q_len,),         np.dtype[np_bf16]]
    kv_stream_ty = np.ndarray[(n_q_total * n_kv * kv_len,), np.dtype[np_bf16]]
    o_stream_ty  = np.ndarray[(n_q_total * o_len,),         np.dtype[np_bf16]]

    block_k    = Kernel("attn_block",    obj_name, [q_tile_ty, kv_tile_ty])
    finalise_k = Kernel("attn_finalise", obj_name, [o_tile_ty])

    fifo_q  = ObjectFifo(q_mem_ty,  name="inQ")
    fifo_kv = ObjectFifo(kv_mem_ty, name="inKV")
    fifo_o  = ObjectFifo(o_mem_ty,  name="outO")

    q_per_core = fifo_q.cons().split(
        offsets=[q_len * i for i in range(n_cores)],
        obj_types=[q_tile_ty] * n_cores,
    )
    kv_per_core = fifo_kv.cons().split(
        offsets=[kv_len * i for i in range(n_cores)],
        obj_types=[kv_tile_ty] * n_cores,
    )
    o_per_core = fifo_o.prod().join(
        offsets=[o_len * i for i in range(n_cores)],
        obj_types=[o_tile_ty] * n_cores,
    )

    def core_fn(of_q, of_kv, of_o, attn_block, attn_finalise):
        for _ in range_(n_per_core) if n_per_core > 1 else range(1):
            elem_o = of_o.acquire(1)
            elem_q = of_q.acquire(1)
            for _ in range_(n_kv) if n_kv > 1 else range(1):
                elem_kv = of_kv.acquire(1)
                attn_block(elem_q, elem_kv)
                of_kv.release(1)
            attn_finalise(elem_o)
            of_q.release(1)
            of_o.release(1)

    workers = []
    for i in range(n_cores):
        workers.append(Worker(
            core_fn,
            fn_args=[q_per_core[i].cons(), kv_per_core[i].cons(),
                     o_per_core[i].prod(), block_k, finalise_k],
            stack_size=0x3000,
        ))

    rt = Runtime()
    with rt.sequence(q_stream_ty, kv_stream_ty, o_stream_ty) as (q, kv, o):
        rt.start(*workers)
        rt.fill(fifo_q.prod(),  q)
        rt.fill(fifo_kv.prod(), kv)
        rt.drain(fifo_o.cons(), o, wait=True)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    out_path.write_text(str(module))


@dataclass
class Compiled:
    BR: int; BC: int; D: int; n_kv: int; n_q_total: int; n_cores: int
    xclbin_path: Path
    insts: np.ndarray


def build_xclbin(BR: int, BC: int, D: int, n_kv: int, n_q_total: int,
                 n_cores: int) -> Compiled:
    tag = f"fa_{BR}x{BC}x{D}_n{n_kv}_q{n_q_total}_c{n_cores}"
    build = CACHE / tag
    build.mkdir(parents=True, exist_ok=True)
    xclbin = build / "final.xclbin"
    insts  = build / "insts.bin"
    if not xclbin.exists() or not insts.exists():
        obj = compile_kernel(BR, BC, D, build)
        mlir = build / "aie.mlir"
        generate_mlir(BR, BC, D, n_kv, n_q_total, n_cores, obj.name, mlir)
        compile_xclbin(mlir, obj, build)
    insts_arr = np.fromfile(insts, dtype=np.uint32)
    return Compiled(BR=BR, BC=BC, D=D, n_kv=n_kv, n_q_total=n_q_total,
                    n_cores=n_cores, xclbin_path=xclbin, insts=insts_arr)


class NpuAttention:
    """Callable that runs FA on a batch of Q-blocks in one NPU dispatch.

    Q-blocks are split across n_cores compute tiles (default 4) for
    parallel execution. The host pads N up to a multiple of n_cores with
    dummy zero-data blocks whose outputs are discarded. xclbins are keyed
    on (n_kv, n_q_total_padded, n_cores).
    """
    def __init__(self, BR: int = 32, BC: int = 32, D: int = 64, n_cores: int = 4):
        self.BR, self.BC, self.D = BR, BC, D
        self.n_cores = n_cores
        self._compiled: dict[tuple[int, int], Compiled] = {}   # (n_kv, n_q_padded)
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
        K_stack: torch.Tensor,     # [N, TK, D]
        V_stack: torch.Tensor,     # [N, TK, D]
        start_rows: list[int],     # length N
        causal: bool = True,
    ) -> torch.Tensor:
        """Return O_stack [N, BR, D] bf16. TK must be a multiple of BC."""
        import pyxrt
        N_in, BR, D = Q_stack.shape
        TK = K_stack.shape[1]
        BC = self.BC
        assert (BR, D) == (self.BR, self.D)
        assert K_stack.shape == (N_in, TK, D) and V_stack.shape == (N_in, TK, D)
        assert TK % BC == 0
        assert len(start_rows) == N_in
        n_kv = TK // BC
        n_cores = self.n_cores

        # Pad N up to a multiple of n_cores with dummy Q-blocks. Dummy rows
        # have start_row set to a very negative value so the causal mask
        # zeroes all their output; the output slice discards these rows.
        N_pad = ((N_in + n_cores - 1) // n_cores) * n_cores
        if N_pad != N_in:
            pad = N_pad - N_in
            Q_stack = torch.cat([Q_stack, Q_stack.new_zeros(pad, BR, D)], dim=0)
            K_stack = torch.cat([K_stack, K_stack.new_zeros(pad, TK, D)], dim=0)
            V_stack = torch.cat([V_stack, V_stack.new_zeros(pad, TK, D)], dim=0)
            start_rows = list(start_rows) + [-1_000_000] * pad  # well past any real col
        N = N_pad
        n_per_core = N // n_cores

        key = (n_kv, N)
        if key not in self._compiled:
            self._compiled[key] = build_xclbin(BR, BC, D, n_kv, N, n_cores)
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

        R_MAC, S_MAC, T_MAC = 4, 8, 8
        MR, MK, MT = BR // R_MAC, D // S_MAC, BC // T_MAC

        Q_bf = Q_stack.to(torch.bfloat16).contiguous()
        Q_tiled = Q_bf.view(N, MR, R_MAC, MK, S_MAC).permute(0, 1, 3, 2, 4).contiguous()

        headers = torch.zeros((N, HEADER_BF16), dtype=torch.bfloat16)
        headers[:, 0] = torch.tensor(start_rows, dtype=torch.bfloat16)
        headers[:, 1] = 1.0 if causal else 0.0
        q_tiles = torch.cat([headers, Q_tiled.reshape(N, -1)], dim=1)  # [N, q_tile_len]

        # Memtile layout: for each of n_per_core outer iterations, stack
        # n_cores tiles contiguously. Flat block f = iter*n_cores + core.
        # q_tiles is already ordered by f (f=0 first), so reshape-and-permute
        # isn't needed — [N, q_tile_len] flattened is exactly the layout
        # the memtile expects (iter 0: [t0, t1, ..., t_{nc-1}], then iter 1, ...).
        q_arr = _bf16_to_u16(q_tiles.reshape(-1)).reshape(-1)
        bo_q.write(q_arr.tobytes(), 0)
        bo_q.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        K_bf = K_stack.to(torch.bfloat16).contiguous().view(N, n_kv, MT, T_MAC, MK, S_MAC)
        K_tiled = K_bf.permute(0, 1, 4, 2, 5, 3).contiguous()  # [N, n_kv, MK, MT, S, T]
        V_bf = V_stack.to(torch.bfloat16).contiguous().view(N, n_kv, BC, D)

        # KV memtile layout: for each outer iter and each kv_iter (inner),
        # stack n_cores KV tiles. So the flat order is
        #   [iter_0 kv_0 core_0, ..., iter_0 kv_0 core_{nc-1},
        #    iter_0 kv_1 core_0, ..., iter_0 kv_{nkv-1} core_{nc-1},
        #    iter_1 kv_0 core_0, ...].
        # Flat block f = iter * nc + core; for (iter, kv, core) we pick
        # KV[f, kv]. Reshape+permute gets us there.
        kv_tile_flat_bytes = K_tiled.reshape(N, n_kv, -1)
        kv_tile_flat_v = V_bf.reshape(N, n_kv, -1)
        # Shape [N, n_kv, 2, BC*D] (K and V interleaved per tile)
        kv_pairs = torch.stack([kv_tile_flat_bytes, kv_tile_flat_v], dim=2)
        # View [n_per_core, n_cores, n_kv, 2, BC*D] then permute to
        # [n_per_core, n_kv, n_cores, 2, BC*D] so the innermost fast axis
        # is (2, BC*D) = one KV tile.
        kv_pairs = (kv_pairs.view(n_per_core, n_cores, n_kv, 2, BC * D)
                            .permute(0, 2, 1, 3, 4).contiguous())
        kv_arr = _bf16_to_u16(kv_pairs.reshape(-1)).reshape(-1)
        bo_kv.write(kv_arr.tobytes(), 0)
        bo_kv.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        run = kernel(3, bo_instr, c.insts.size, bo_q, bo_kv, bo_o)
        run.wait()

        bo_o.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out = np.frombuffer(bytes(bo_o.read(o_bytes, 0)), dtype=np.uint16).copy()
        out_t = torch.from_numpy(out).view(torch.bfloat16).reshape(N, BR, D)
        return out_t[:N_in]  # discard padding rows

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
