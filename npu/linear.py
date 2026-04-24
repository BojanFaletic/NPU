"""NpuLinear: torch-callable linear layer backed by XDNA 2 bf16 matmul.

Caches one compiled xclbin per (M, K, N) shape on disk.  Pads M and N to
multi-core-divisible boundaries and slices the result back at the end.
"""
from __future__ import annotations
import os, sys, hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from matmul import compile_kernel, compile_xclbin, _bf16_to_u16

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "mlir-aie-src" / "programming_examples" / "basic" / "matrix_multiplication" / "whole_array"))
import whole_array_iron as wai

sys.path.insert(0, "/opt/xilinx/xrt/python")
import pyxrt


ROOT = Path(__file__).parent
CACHE = ROOT / "build" / "linear_cache"


# -------------------- shape planner --------------------

@dataclass
class Plan:
    M: int; K: int; N: int          # actual requested shape
    m: int; k: int; n: int          # per-core tile
    cols: int                        # n_aie_cols
    M_pad: int; N_pad: int           # padded outer dims

    @classmethod
    def make(cls, M: int, K: int, N: int, force_N_pad: int | None = None) -> "Plan":
        """Pick tile + col count that fit our constraints, minimise padding.

        When `force_N_pad` is given, round N_pad up to it (must be >= N and
        satisfy the N_pad % (n*cols) divisibility). This lets multiple
        NpuLinears with different out-features share a single xclbin: the
        driver avoids an expensive hw_context/kernel switch between them, at
        the cost of some padded compute on the narrower weights. The concrete
        trade-off is tested empirically — with 4 SmolLM projection shapes a
        round-robin call pattern costs 2.56× more per call than single-shape,
        so folding three K=576 projections into one shared (K=576, N=3072)
        xclbin is a net win even with 3-5× compute inflation on wqkv/wo.
        """
        # Constraints (whole_array_iron.py):
        #   M_pad must divide evenly into (m * n_aie_rows * tb_n_rows) blocks
        #     where n_aie_rows=4, tb_n_rows=tb_max_n_rows//2=2  =>  M_step = m*8
        #   K   % k                  == 0
        #   N_pad % (n * cols)       == 0
        #   m%r==0, k%s==0, n%t==0  (r,s,t)=(4,8,8) for bf16 on npu2
        m, k, n = 32, 64, 32
        best: Plan | None = None
        for cols in (8, 4, 2, 1):
            N_step = n * cols
            if force_N_pad is not None:
                if force_N_pad < N or force_N_pad % N_step != 0:
                    continue
                N_pad = force_N_pad
            else:
                N_pad = (N + N_step - 1) // N_step * N_step
            M_step = m * 4 * 2  # m * n_aie_rows * tb_n_rows
            M_pad = (M + M_step - 1) // M_step * M_step
            if K % k != 0:
                continue
            score = M_pad * N_pad
            if best is None or score < best.M_pad * best.N_pad or (score == best.M_pad * best.N_pad and cols > best.cols):
                best = cls(M=M, K=K, N=N, m=m, k=k, n=n, cols=cols, M_pad=M_pad, N_pad=N_pad)
        assert best is not None, (
            f"no valid plan for shape M={M} K={K} N={N} force_N_pad={force_N_pad} "
            f"(K must be multiple of 64; force_N_pad must be reachable by some "
            f"(n*cols) divisor)"
        )
        return best


# -------------------- per-shape compiled xclbin --------------------

@dataclass
class Compiled:
    plan: Plan
    xclbin_path: Path
    insts: np.ndarray


def _build_xclbin(plan: Plan) -> Compiled:
    tag = f"mm_{plan.M_pad}x{plan.K}x{plan.N_pad}_{plan.m}x{plan.k}x{plan.n}_c{plan.cols}"
    build = CACHE / tag
    build.mkdir(parents=True, exist_ok=True)
    xclbin = build / "final.xclbin"
    insts  = build / "insts.bin"
    if not xclbin.exists() or not insts.exists():
        obj = compile_kernel(plan.m, plan.k, plan.n, build)
        mlir = build / "aie.mlir"
        module = wai.my_matmul(
            dev="npu2",
            M=plan.M_pad, K=plan.K, N=plan.N_pad,
            m=plan.m, k=plan.k, n=plan.n,
            n_aie_cols=plan.cols,
            dtype_in_str="bf16", dtype_out_str="f32",
            b_col_maj=0, emulate_bf16_mmul_with_bfp16=False,
            trace_size=0, generate_taps=False,
        )
        mlir.write_text(str(module))
        compile_xclbin(mlir, obj, build)
    insts_arr = np.fromfile(insts, dtype=np.uint32)
    return Compiled(plan=plan, xclbin_path=xclbin, insts=insts_arr)


# -------------------- device / xclbin registry --------------------

class _XrtCtx:
    """Lazy singleton device + per-xclbin kernel cache."""
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
        xb  = pyxrt.xclbin(str(c.xclbin_path))
        kname = next(k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE"))
        uuid = dev.register_xclbin(xb)
        ctx = pyxrt.hw_context(dev, uuid)
        kernel = pyxrt.kernel(ctx, kname)
        bo_instr = pyxrt.bo(dev, c.insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
        bo_instr.write(c.insts.tobytes(), 0)
        bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        cls._kernels[c.xclbin_path] = (xb, ctx, kernel, bo_instr)
        return cls._kernels[c.xclbin_path]


# -------------------- the linear layer --------------------

class NpuLinear:
    """torch.nn.functional.linear(x, W) — but runs W.T matmul on the NPU.

    Equivalent to `x @ W.T` in math. The weight is pre-transposed once at
    construction time. Supports only 2-D inputs internally; any extra leading
    batch/time dims are flattened before dispatch and un-flattened after.
    """

    def __init__(self, weight: torch.Tensor, share_N_pad: int | None = None):
        """weight: [out, in]. share_N_pad: if set, force the xclbin's padded
        N dim to this value so multiple NpuLinears with different out_features
        (but same K) compile to / reuse the same xclbin. See Plan.make."""
        assert weight.dim() == 2, "expected [out, in]"
        self.out_features, self.in_features = weight.shape
        self.share_N_pad = share_N_pad
        # bf16 weight, transposed so dispatched matmul is A @ B (not A @ B.T).
        self._W = weight.to(torch.bfloat16).t().contiguous()  # shape [in, out]
        # Per-M compiled kernel + preallocated device buffers.
        self._compiled: dict[int, Compiled] = {}
        # Per-M activation + output buffers; weight buffer is per-instance (one).
        self._bo_a: dict[int, pyxrt.bo] = {}
        self._bo_c: dict[int, pyxrt.bo] = {}
        self._bo_b: dict[Path, pyxrt.bo] = {}        # keyed by xclbin path
        self._B_pad_cache: dict[int, torch.Tensor] = {}  # padded B per N_pad

    def _plan_for(self, M: int) -> Compiled:
        if M not in self._compiled:
            plan = Plan.make(M=M, K=self.in_features, N=self.out_features,
                             force_N_pad=self.share_N_pad)
            self._compiled[M] = _build_xclbin(plan)
        return self._compiled[M]

    def _get_B_pad(self, N_pad: int) -> torch.Tensor:
        cached = self._B_pad_cache.get(N_pad)
        if cached is not None:
            return cached
        if N_pad != self.out_features:
            B = torch.cat(
                [self._W, torch.zeros(self.in_features, N_pad - self.out_features, dtype=torch.bfloat16)],
                dim=1,
            ).contiguous()
        else:
            B = self._W
        self._B_pad_cache[N_pad] = B
        return B

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from npu.profiler import profile  # same module instance as bench/smollm
        orig_shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        M = x2.shape[0]
        c = self._plan_for(M)
        plan = c.plan

        dev = _XrtCtx.device()
        _, _, kernel, bo_instr = _XrtCtx.kernel_for(c)

        # --- weight buffer: upload once per xclbin, reused forever ---
        bo_b = self._bo_b.get(c.xclbin_path)
        if bo_b is None:
            B_bf = self._get_B_pad(plan.N_pad)
            b_np = _bf16_to_u16(B_bf).reshape(-1)
            bo_b = pyxrt.bo(dev, b_np.nbytes, pyxrt.bo.host_only, kernel.group_id(4))
            bo_b.write(b_np.tobytes(), 0)
            bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._bo_b[c.xclbin_path] = bo_b

        # --- activation + output buffers: one per M, reused across calls ---
        bo_a = self._bo_a.get(M)
        bo_c = self._bo_c.get(M)
        if bo_a is None:
            bo_a = pyxrt.bo(dev, plan.M_pad * plan.K * 2,    pyxrt.bo.host_only, kernel.group_id(3))
            bo_c = pyxrt.bo(dev, plan.M_pad * plan.N_pad * 4, pyxrt.bo.host_only, kernel.group_id(5))
            self._bo_a[M] = bo_a
            self._bo_c[M] = bo_c

        with profile("lin.bf16"):
            A_bf = x2.to(torch.bfloat16).contiguous()
            if plan.M_pad != M:
                A_bf = torch.cat(
                    [A_bf, torch.zeros(plan.M_pad - M, plan.K, dtype=torch.bfloat16)], dim=0)
            a_np = _bf16_to_u16(A_bf).reshape(-1)
        with profile("lin.write"):
            bo_a.write(a_np.tobytes(), 0)
        with profile("lin.sync_to"):
            bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        with profile("lin.kernel"):
            run = kernel(3, bo_instr, c.insts.size, bo_a, bo_b, bo_c)
            run.wait()
        with profile("lin.sync_from"):
            bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        with profile("lin.read"):
            c_out = np.frombuffer(bytes(bo_c.read(plan.M_pad * plan.N_pad * 4, 0)), dtype=np.float32)
            c_out = c_out.reshape(plan.M_pad, plan.N_pad)[:M, :self.out_features].copy()
        with profile("lin.to_torch"):
            return torch.from_numpy(c_out).reshape(*orig_shape[:-1], self.out_features)


# -------------------- self-test --------------------

def _self_test():
    torch.manual_seed(0)
    # shape matching SmolLM2 Q projection with a 256-token prefill
    in_f, out_f = 576, 576
    W = torch.randn(out_f, in_f, dtype=torch.bfloat16)
    x = torch.randn(1, 256, in_f, dtype=torch.bfloat16)

    lin = NpuLinear(W)
    y_npu = lin(x)
    y_ref = torch.nn.functional.linear(x.float(), W.float())

    diff = (y_npu - y_ref).abs()
    print(f"NpuLinear [{x.shape}] x W[{W.shape}] : max|Δ|={diff.max():.3e}  mean|Δ|={diff.mean():.3e}")

    assert diff.max() < 5e-2, "output deviates too much"
    print("OK")


if __name__ == "__main__":
    _self_test()
