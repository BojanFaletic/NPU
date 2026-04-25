"""End-to-end NPU smoke test: prompt prefill + 5 generated tokens.

This is the integration guard to run before landing major model/NPU changes.
It checks:
  - NPU generated token IDs and logits against a selectable baseline,
  - current and peak process RSS at major stages.

Default baseline is the cached llama.cpp oracle from qwen/ref_cache, so normal
benchmark runs skip the slow Python CPU pass. Use `--full-check` before larger
model/NPU integrations to run the Python CPU baseline and require llama
agreement too. The llama check is margin-aware: near-tied top-1 disagreements
are reported as ambiguous instead of structural failures, and logits after an
ambiguous branch are not compared because the sequence context has diverged.

Run:
    PYTHONPATH=. uv run python qwen/test_npu_generate5.py
"""
from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwen.forward import Model, enable_npu, forward
from qwen.model import TensorStore


DEFAULT_NPU_OPS = "router,shexp,experts,attn_o,attn_qkv,ssm"
GGUF = ROOT / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
CACHE = ROOT / "qwen" / "ref_cache"


def rss_mib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def mark(label: str, t0: float | None = None) -> None:
    elapsed = "" if t0 is None else f"  dt={time.time() - t0:.2f}s"
    print(
        f"mem {label:<18s} rss={rss_mib():8.1f} MiB  "
        f"peak={peak_rss_mib():8.1f} MiB{elapsed}"
    )


def compare_logits(
    label: str,
    pos: int,
    got: torch.Tensor,
    ref: torch.Tensor | np.ndarray,
    *,
    cos_min: float,
    max_abs_limit: float | None,
    top_margin: float | None = None,
) -> bool:
    got_np = got.detach().float().reshape(-1).numpy()
    if isinstance(ref, torch.Tensor):
        ref_np = ref.detach().float().reshape(-1).numpy()
    else:
        ref_np = ref.reshape(-1).astype(np.float32, copy=False)
    diff = got_np - ref_np
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    denom = float(np.linalg.norm(got_np) * np.linalg.norm(ref_np))
    cos = float(np.dot(got_np, ref_np) / max(denom, 1e-20))
    top1 = int(got_np.argmax())
    ref_top1 = int(ref_np.argmax())
    top_ok = top1 == ref_top1
    ambiguous = False
    if not top_ok and top_margin is not None and top_margin > 0:
        ref_gap = float(ref_np[ref_top1] - ref_np[top1])
        got_gap = float(got_np[top1] - got_np[ref_top1])
        ambiguous = ref_gap >= 0 and got_gap >= 0 and max(ref_gap, got_gap) <= top_margin
        top_ok = ambiguous
    ok = top_ok and cos >= cos_min
    if max_abs_limit is not None:
        ok = ok and max_abs <= max_abs_limit
    status = "OK" if ok and not ambiguous else "AMBIG" if ok else "FAIL"
    print(
        f"{label} logits pos={pos:02d} top1={top1:>6d}/{ref_top1:<6d} "
        f"cos={cos:.6f} max|d|={max_abs:.3e} mean|d|={mean_abs:.3e} "
        f"{status}"
    )
    return ok


def top_mismatch_is_ambiguous(
    got: torch.Tensor,
    ref: np.ndarray,
    *,
    top_margin: float,
) -> bool:
    if top_margin <= 0:
        return False
    got_np = got.detach().float().reshape(-1).numpy()
    ref_np = ref.reshape(-1).astype(np.float32, copy=False)
    got_top = int(got_np.argmax())
    ref_top = int(ref_np.argmax())
    if got_top == ref_top:
        return False
    ref_gap = float(ref_np[ref_top] - ref_np[got_top])
    got_gap = float(got_np[got_top] - got_np[ref_top])
    return ref_gap >= 0 and got_gap >= 0 and max(ref_gap, got_gap) <= top_margin


def compare_oracle_trace(
    label: str,
    trace: "Trace",
    prompt_len: int,
    oracle_top1: np.ndarray,
    oracle_logits: np.ndarray,
    *,
    n_gen: int,
    cos_min: float,
    max_abs_limit: float | None,
    top_margin: float,
) -> bool:
    """Compare generation-critical rows against cached llama.cpp outputs.

    Row prompt_len - 1 chooses the first generated token. Each later generated
    token is chosen by the previous decode row. Once a near-tied top-1 branch
    differs, later rows are for different token histories and are skipped.
    """
    ok = True
    max_steps = min(
        n_gen,
        len(trace.generated),
        len(oracle_top1) - (prompt_len - 1),
        len(trace.logits) - (prompt_len - 1),
    )
    branch_pos: int | None = None
    got_tokens: list[int] = []
    ref_tokens: list[int] = []

    for step in range(max_steps):
        pos = prompt_len + step - 1
        got = trace.generated[step]
        ref = int(oracle_top1[pos])
        got_tokens.append(got)
        ref_tokens.append(ref)
        if got == ref:
            print(f"{label} token step={step} pos={pos:02d} tok={got} OK")
            continue
        if top_mismatch_is_ambiguous(
            trace.logits[pos], oracle_logits[pos], top_margin=top_margin,
        ):
            print(
                f"{label} token step={step} pos={pos:02d} "
                f"tok={got}/{ref} AMBIG"
            )
            branch_pos = pos
            break
        print(
            f"{label} token step={step} pos={pos:02d} "
            f"tok={got}/{ref} FAIL"
        )
        ok = False
        branch_pos = pos
        break

    status = "OK" if ok and branch_pos is None else "AMBIG" if ok else "FAIL"
    print(f"{label} tokens got={got_tokens} ref={ref_tokens} {status}")

    last_pos = branch_pos if branch_pos is not None else prompt_len + max_steps - 2
    for pos in range(prompt_len - 1, last_pos + 1):
        ok = compare_logits(
            label, pos, trace.logits[pos], oracle_logits[pos],
            cos_min=cos_min,
            max_abs_limit=max_abs_limit,
            top_margin=top_margin,
        ) and ok
    if branch_pos is not None and ok:
        print(f"{label} logits after pos={branch_pos:02d} skipped after ambiguous branch")
    return ok


@dataclass
class Trace:
    generated: list[int]
    logits: list[torch.Tensor]


def run_generate(
    label: str,
    model: Model,
    prompt: list[int],
    n_gen: int,
) -> Trace:
    rows: list[torch.Tensor] = []
    generated: list[int] = []

    t0 = time.time()
    logits, kvs, ssms = forward(model, prompt, start_pos=0)
    mark(f"{label}.prefill", t0)
    for row in logits:
        rows.append(row.detach().float().cpu().clone())

    prev_logits = rows[-1]
    for step in range(n_gen):
        nxt = int(prev_logits.argmax().item())
        generated.append(nxt)
        print(f"{label} gen step={step} tok={nxt}")

        # Generating N tokens only requires N-1 decode forwards after the
        # prefill logits. The final token is validated as an ID, without asking
        # the model for an uncached next-logit row.
        if step + 1 == n_gen:
            break

        start = len(prompt) + step
        t0 = time.time()
        logits, kvs, ssms = forward(
            model, [nxt], start_pos=start,
            kv_caches=kvs, ssm_states=ssms,
        )
        mark(f"{label}.decode.{step}", t0)
        prev_logits = logits[-1].detach().float().cpu().clone()
        rows.append(prev_logits)

    return Trace(generated=generated, logits=rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(GGUF))
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--n-gen", type=int, default=5)
    ap.add_argument("--max-pos", type=int, default=512)
    ap.add_argument("--npu", default=DEFAULT_NPU_OPS)
    ap.add_argument("--baseline", choices=("llama", "cpu"), default="llama",
                    help="baseline used for pass/fail checks; default skips "
                         "the slow Python CPU generation pass")
    ap.add_argument("--expert-cache-limit", type=int, default=None)
    ap.add_argument("--logit-cos-min", type=float, default=0.999)
    ap.add_argument("--max-logit-abs", type=float, default=None)
    ap.add_argument("--oracle-top-margin", type=float, default=0.05,
                    help="llama top-1 mismatches within this two-way logit "
                         "margin are reported as ambiguous instead of failing; "
                         "set 0 for strict top-1")
    ap.add_argument("--max-rss-gib", type=float, default=24.0)
    ap.add_argument("--require-oracle", action="store_true",
                    help="in --baseline cpu mode, also fail on llama.cpp drift")
    ap.add_argument("--full-check", action="store_true",
                    help="run the slower CPU baseline and require cached "
                         "llama.cpp agreement")
    args = ap.parse_args()
    if args.full_check:
        args.baseline = "cpu"
        args.require_oracle = True

    cache = Path(args.cache)
    prompt = np.load(cache / "prompt_tokens.npy").astype(np.int64).tolist()
    oracle_top1 = np.load(cache / "top1.npy").astype(np.int64)
    oracle_logits = np.load(cache / "logits.npy")
    torch.set_grad_enabled(False)
    ok = True

    t_all = time.time()
    mark("start")
    t0 = time.time()
    ts = TensorStore(args.model)
    mark("tensorstore", t0)

    t0 = time.time()
    model = Model.load(ts, max_pos=args.max_pos)
    mark("model.load", t0)

    ops = tuple(s.strip() for s in args.npu.split(",") if s.strip())

    with torch.inference_mode():
        oracle_token_count = min(args.n_gen, len(oracle_top1) - (len(prompt) - 1))

        if args.baseline == "cpu":
            cpu = run_generate("cpu", model, prompt, args.n_gen)

            oracle_ok = compare_oracle_trace(
                "llama/cpu", cpu, len(prompt), oracle_top1, oracle_logits,
                n_gen=oracle_token_count,
                cos_min=args.logit_cos_min,
                max_abs_limit=args.max_logit_abs,
                top_margin=args.oracle_top_margin,
            )
            if args.require_oracle:
                ok = oracle_ok and ok

            gc.collect()
            mark("after_cpu")
        else:
            cpu = None

        if ops:
            t0 = time.time()
            enable_npu(model, ops=ops, expert_cache_limit=args.expert_cache_limit)
            mark("enable_npu", t0)
        npu = run_generate("npu", model, prompt, args.n_gen)

        if args.baseline == "llama":
            ok = compare_oracle_trace(
                "llama/npu", npu, len(prompt), oracle_top1, oracle_logits,
                n_gen=oracle_token_count,
                cos_min=args.logit_cos_min,
                max_abs_limit=args.max_logit_abs,
                top_margin=args.oracle_top_margin,
            ) and ok
        else:
            assert cpu is not None
            token_ok = npu.generated == cpu.generated
            ok = token_ok and ok
            print(
                f"npu/cpu tokens npu={npu.generated} cpu={cpu.generated} "
                f"{'OK' if token_ok else 'FAIL'}"
            )
            if len(npu.logits) != len(cpu.logits):
                print(f"npu/cpu logits row count {len(npu.logits)}/{len(cpu.logits)} FAIL")
                ok = False
            for pos, (npu_row, cpu_row) in enumerate(zip(npu.logits, cpu.logits)):
                ok = compare_logits(
                    "npu/cpu", pos, npu_row, cpu_row,
                    cos_min=args.logit_cos_min,
                    max_abs_limit=args.max_logit_abs,
                ) and ok

    peak_gib = peak_rss_mib() / 1024.0
    mem_ok = args.max_rss_gib <= 0 or peak_gib <= args.max_rss_gib
    print(
        f"memory peak={peak_gib:.2f} GiB "
        f"limit={args.max_rss_gib:.2f} GiB {'OK' if mem_ok else 'FAIL'}"
    )
    print(f"total time={time.time() - t_all:.2f}s")
    return 0 if ok and mem_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
