"""Chatbot-shaped benchmark: prefill + decode at log-spaced context lengths.

A real chatbot's cost is dominated by two things:
  1. Time-to-first-token (TTFT)  — prefill of the full prompt.
  2. Inter-token latency (ITL)   — one decode step with growing KV cache.

So at each context length L we measure:
  - prefill(L)          : wall time to process L tokens in one forward.
  - decode_step(L)      : wall time for one extra forward with ctx already at L.
  - decode_N(L, N)      : sustained tok/s over N generated tokens starting at L.
  - correctness(L)      : top-1 token (+ max|Δ| logits) vs HuggingFace reference.

Log-spaced L covers the full usable range from short system prompts to near-
max-context. Correctness runs first at every L — a speed improvement that
breaks the model is worse than no improvement.
"""
from __future__ import annotations
import argparse, math, sys, time
from dataclasses import dataclass
import torch

from smollm import load
from npu.profiler import PROF


DEFAULT_LENGTHS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _ids_for(L: int, vocab: int) -> torch.Tensor:
    """Deterministic synthetic prompt of length L."""
    torch.manual_seed(12345 + L)
    return torch.randint(0, min(vocab, 32000), (1, L))


def _time_fn(fn, warmup: int, iters: int) -> tuple[float, float]:
    """Return (median_s, min_s) wall time over `iters` runs after `warmup`."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


@dataclass
class Row:
    L: int
    cpu_prefill_ms: float
    npu_prefill_ms: float
    cpu_step_ms:    float
    npu_step_ms:    float
    cpu_decode_toks: float
    npu_decode_toks: float
    top1_match: bool
    max_logit_delta: float


def bench_len(
    model_cpu, model_npu, hf,
    L: int, decode_n: int, warmup: int, iters: int,
) -> Row:
    cfg = model_cpu.cfg
    if L > cfg.max_pos:
        raise ValueError(f"L={L} exceeds max_pos={cfg.max_pos}")

    ids = _ids_for(L, cfg.vocab)

    # --- correctness: HF vs NPU top-1 at position L-1 ---
    with torch.no_grad():
        hf_logits = hf(ids).logits[0, -1]
        npu_logits, _ = model_npu.forward(ids)
        npu_logits = npu_logits[0, -1]
    top1_match = hf_logits.argmax().item() == npu_logits.argmax().item()
    max_delta  = (hf_logits - npu_logits).abs().max().item()

    # --- prefill latency at ctx 0 → L ---
    def cpu_prefill():
        with torch.no_grad():
            model_cpu.forward(ids)
    def npu_prefill():
        with torch.no_grad():
            model_npu.forward(ids)
    cpu_p, _ = _time_fn(cpu_prefill, warmup, iters)
    npu_p, _ = _time_fn(npu_prefill, warmup, iters)

    # --- decode-step latency with ctx already == L ---
    with torch.no_grad():
        _, cpu_cache = model_cpu.forward(ids)
        _, npu_cache = model_npu.forward(ids)
    next_id = torch.tensor([[ids[0, -1].item()]])

    def cpu_step():
        with torch.no_grad():
            model_cpu.forward(next_id, cache=cpu_cache, start_pos=L)
    def npu_step():
        with torch.no_grad():
            model_npu.forward(next_id, cache=npu_cache, start_pos=L)
    cpu_s, _ = _time_fn(cpu_step, max(1, warmup // 2), iters)
    npu_s, _ = _time_fn(npu_step, max(1, warmup // 2), iters)

    # --- sustained decode tok/s over N new tokens ---
    def cpu_decode():
        nonlocal_cache = [(k.clone(), v.clone()) for (k, v) in cpu_cache]
        nid = next_id.clone()
        pos = L
        with torch.no_grad():
            for _ in range(decode_n):
                logits, nonlocal_cache = model_cpu.forward(nid, cache=nonlocal_cache, start_pos=pos)
                nid = logits[:, -1, :].argmax(-1, keepdim=True)
                pos += 1
    def npu_decode():
        nonlocal_cache = [(k.clone(), v.clone()) for (k, v) in npu_cache]
        nid = next_id.clone()
        pos = L
        with torch.no_grad():
            for _ in range(decode_n):
                logits, nonlocal_cache = model_npu.forward(nid, cache=nonlocal_cache, start_pos=pos)
                nid = logits[:, -1, :].argmax(-1, keepdim=True)
                pos += 1
    cpu_d, _ = _time_fn(cpu_decode, 1, max(1, iters // 2))
    npu_d, _ = _time_fn(npu_decode, 1, max(1, iters // 2))

    return Row(
        L=L,
        cpu_prefill_ms=cpu_p * 1e3,
        npu_prefill_ms=npu_p * 1e3,
        cpu_step_ms=cpu_s * 1e3,
        npu_step_ms=npu_s * 1e3,
        cpu_decode_toks=decode_n / cpu_d,
        npu_decode_toks=decode_n / npu_d,
        top1_match=top1_match,
        max_logit_delta=max_delta,
    )


def _print_header():
    print()
    print(f"{'L':>6} "
          f"{'prefill ms':>22} "
          f"{'step ms':>18} "
          f"{'decode tok/s':>22} "
          f"{'top1':>5} {'max|Δ|':>10}")
    print(f"{'':>6} "
          f"{'cpu':>10} {'npu':>10} "
          f"{'cpu':>8} {'npu':>8} "
          f"{'cpu':>10} {'npu':>10} ")


def _print_row(r: Row, skipped: bool = False):
    if skipped:
        print(f"{r.L:>6}  (skipped)")
        return
    ok = "ok" if r.top1_match else "MISS"
    print(f"{r.L:>6} "
          f"{r.cpu_prefill_ms:>10.1f} {r.npu_prefill_ms:>10.1f} "
          f"{r.cpu_step_ms:>8.2f} {r.npu_step_ms:>8.2f} "
          f"{r.cpu_decode_toks:>10.2f} {r.npu_decode_toks:>10.2f} "
          f"{ok:>5} {r.max_logit_delta:>10.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", default=",".join(str(L) for L in DEFAULT_LENGTHS),
                    help="comma-separated context lengths")
    ap.add_argument("--decode-n", type=int, default=16,
                    help="tokens to generate for sustained tok/s (default 16)")
    ap.add_argument("--warmup",   type=int, default=2)
    ap.add_argument("--iters",    type=int, default=3)
    ap.add_argument("--no-npu",   action="store_true", help="skip the NPU model (debug)")
    ap.add_argument("--profile",  action="store_true",
                    help="print per-op timing breakdown after the sweep")
    ap.add_argument("--max-L",    type=int, default=None,
                    help="cap max length (default: cfg.max_pos)")
    ap.add_argument("--cpu-decode-fallback", action="store_true",
                    help="with NPU enabled, route single-token decode projections to CPU")
    ap.add_argument("--npu-decode-attn", action="store_true",
                    help="with NPU enabled, route single-token attention through NpuAttention")
    ap.add_argument("--npu-decode-matvec", action="store_true",
                    help="with NPU enabled, route single-token projections through NpuMatVec")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    print("loading CPU model…")
    model_cpu, tok, hf = load(torch.float32)
    cap = args.max_L if args.max_L is not None else model_cpu.cfg.max_pos
    lengths = [L for L in lengths if L <= cap]
    print(f"lengths: {lengths}  (cfg.max_pos={model_cpu.cfg.max_pos}, cap={cap})")

    if args.no_npu:
        model_npu = model_cpu
    else:
        print("loading NPU model (first forward at each new L will compile xclbins)…")
        model_npu, _, _ = load(torch.float32)
        model_npu.enable_npu(
            cpu_decode_fallback=args.cpu_decode_fallback,
            decode_attention=args.npu_decode_attn,
            decode_matvec=args.npu_decode_matvec,
        )

    # Pre-warm: trigger xclbin compile at every L so the reported prefill
    # time excludes cold-compile cost. Prints a per-L compile time.
    if not args.no_npu:
        print("\npre-warming NPU xclbins:")
        for L in lengths:
            ids = _ids_for(L, model_cpu.cfg.vocab)
            t0 = time.time()
            with torch.no_grad():
                model_npu.forward(ids)
            print(f"  L={L:>5} compile+first-call = {time.time() - t0:6.1f}s")

    if args.profile:
        PROF.enable()

    _print_header()
    rows: list[Row] = []
    for L in lengths:
        try:
            r = bench_len(model_cpu, model_npu, hf, L,
                          decode_n=args.decode_n, warmup=args.warmup, iters=args.iters)
            _print_row(r)
            rows.append(r)
        except Exception as e:
            print(f"{L:>6}  ERROR: {type(e).__name__}: {e}")

    # Correctness summary
    bad = [r for r in rows if not r.top1_match]
    if bad:
        print(f"\nCORRECTNESS: {len(bad)}/{len(rows)} lengths failed top-1 match:")
        for r in bad:
            print(f"  L={r.L}  max|Δ|={r.max_logit_delta:.2e}")
        sys.exit(1)
    else:
        print(f"\nCORRECTNESS: all {len(rows)}/{len(rows)} lengths match HF top-1")

    if args.profile:
        print("\n--- per-op profile (cumulative across sweep) ---")
        PROF.report()


if __name__ == "__main__":
    main()
