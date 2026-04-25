"""End-to-end NPU smoke test: prompt prefill + 5 generated tokens.

This is the integration guard to run before landing major model/NPU changes.
It checks:
  - NPU generated token IDs and logits against a selectable baseline,
  - current and peak process RSS at major stages.

Default baseline is the current Python CPU path. Use `--baseline llama` for the
stricter cached llama.cpp oracle from qwen/ref_cache. That mode currently fails
because the Python model itself drifts after the first generated token; it is
kept as the model-correctness target.

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
    ok = top1 == ref_top1 and cos >= cos_min
    if max_abs_limit is not None:
        ok = ok and max_abs <= max_abs_limit
    print(
        f"{label} logits pos={pos:02d} top1={top1:>6d}/{ref_top1:<6d} "
        f"cos={cos:.6f} max|d|={max_abs:.3e} mean|d|={mean_abs:.3e} "
        f"{'OK' if ok else 'FAIL'}"
    )
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
    ap.add_argument("--baseline", choices=("cpu", "llama"), default="cpu",
                    help="baseline used for pass/fail checks")
    ap.add_argument("--expert-cache-limit", type=int, default=None)
    ap.add_argument("--logit-cos-min", type=float, default=0.999)
    ap.add_argument("--max-logit-abs", type=float, default=None)
    ap.add_argument("--max-rss-gib", type=float, default=24.0)
    ap.add_argument("--require-oracle", action="store_true",
                    help="in --baseline cpu mode, also fail on llama.cpp drift")
    args = ap.parse_args()

    cache = Path(args.cache)
    prompt = np.load(cache / "prompt_tokens.npy").astype(np.int64).tolist()
    oracle_top1 = np.load(cache / "top1.npy").astype(np.int64)
    oracle_logits = np.load(cache / "logits.npy")
    torch.set_grad_enabled(False)
    ok = True

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
        oracle_tokens = [
            int(oracle_top1[len(prompt) + step - 1])
            for step in range(oracle_token_count)
        ]

        if args.baseline == "cpu":
            cpu = run_generate("cpu", model, prompt, args.n_gen)

            oracle_ok = True
            got_tokens = cpu.generated[:oracle_token_count]
            token_ok = got_tokens == oracle_tokens
            oracle_ok = token_ok and oracle_ok
            print(
                f"llama report cpu={got_tokens} ref={oracle_tokens} "
                f"{'OK' if token_ok else 'DRIFT'}"
            )
            for pos, row in enumerate(cpu.logits[:len(oracle_logits)]):
                oracle_ok = compare_logits(
                    "llama", pos, row, oracle_logits[pos],
                    cos_min=args.logit_cos_min,
                    max_abs_limit=args.max_logit_abs,
                ) and oracle_ok
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
            got_tokens = npu.generated[:oracle_token_count]
            token_ok = got_tokens == oracle_tokens
            ok = token_ok and ok
            print(
                f"llama tokens npu={got_tokens} ref={oracle_tokens} "
                f"{'OK' if token_ok else 'FAIL'}"
            )
            for pos, row in enumerate(npu.logits[:len(oracle_logits)]):
                ok = compare_logits(
                    "llama", pos, row, oracle_logits[pos],
                    cos_min=args.logit_cos_min,
                    max_abs_limit=args.max_logit_abs,
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
    return 0 if ok and mem_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
