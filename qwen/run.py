"""End-to-end CPU forward on the real Qwen3.6-35B-A3B weights.

Loads the GGUF, runs our Python forward on the oracle's prompt tokens, and
compares top-1 against qwen/ref_cache/top1.npy (produced by ref_llama.py).

Usage:
    uv run python qwen/run.py                  # use cached prompt tokens
    uv run python qwen/run.py --n-gen 0        # prefill-only gating
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from qwen.model import TensorStore
from qwen.forward import Model, enable_npu, forward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="qwen/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf")
    ap.add_argument("--cache", default="qwen/ref_cache")
    ap.add_argument("--n-gen", type=int, default=0,
                    help="how many tokens to greedy-generate after prefill")
    ap.add_argument("--max-pos", type=int, default=512,
                    help="rope cache size (keep small to save memory)")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="debug: run only the first N layers")
    ap.add_argument("--skip-moe", action="store_true",
                    help="debug: replace MoE output with zeros (fast)")
    ap.add_argument("--trace", action="store_true",
                    help="debug: print per-layer hidden norm")
    ap.add_argument("--n-prompt", type=int, default=None,
                    help="debug: slice prompt to the first N tokens")
    ap.add_argument("--npu", default=None,
                    help="comma-sep ops to dispatch on the XDNA 2 NPU "
                         "(router, shexp, experts, experts_dense, attn_o, "
                         "attn_qkv, ssm). Each op is T=1 only; T>1 "
                         "transparently falls back to F.linear.")
    ap.add_argument("--expert-cache-limit", type=int, default=None,
                    help="max routed experts cached per layer for NPU experts "
                         "(default: 32 compact, 8 dense; -1 = unlimited)")
    args = ap.parse_args()

    cache = Path(args.cache)
    prompt_tokens = np.load(cache / "prompt_tokens.npy").tolist()
    oracle_top1 = np.load(cache / "top1.npy")
    oracle_all = np.load(cache / "all_tokens.npy")
    if args.n_prompt is not None:
        prompt_tokens = prompt_tokens[: args.n_prompt]
    print(f"prompt tokens: {prompt_tokens}")
    print(f"oracle top1 (prefill + gen): {oracle_top1.tolist()}")
    print(f"oracle full sequence:        {oracle_all.tolist()}")

    t0 = time.time()
    ts = TensorStore(args.model)
    print(f"tensor store: {time.time()-t0:.1f}s   "
          f"cfg layers={ts.cfg.n_layer}")

    t0 = time.time()
    model = Model.load(ts, max_pos=args.max_pos)
    print(f"model load:   {time.time()-t0:.1f}s   "
          f"(MoE experts are lazy; dequanted per forward)")

    if args.npu:
        ops = tuple(s.strip() for s in args.npu.split(",") if s.strip())
        t0 = time.time()
        enable_npu(model, ops=ops, expert_cache_limit=args.expert_cache_limit)
        print(f"NPU enabled:  ops={ops}  ({time.time()-t0:.1f}s)")

    # --- Prefill ---
    print(f"\nprefill T={len(prompt_tokens)}  "
          f"n_layers={args.n_layers or ts.cfg.n_layer}  "
          f"skip_moe={args.skip_moe} ...")
    t0 = time.time()
    logits, kvs, ssms = forward(
        model, prompt_tokens, start_pos=0,
        n_layer=args.n_layers, skip_moe=args.skip_moe, trace=args.trace,
    )
    print(f"prefill:      {time.time()-t0:.1f}s   logits={tuple(logits.shape)}")

    ours_top1 = logits.argmax(-1).tolist()
    expected = oracle_top1[:len(prompt_tokens)].tolist()
    matches = sum(1 for a, b in zip(ours_top1, expected) if a == b)
    print(f"\nprefill top-1 match: {matches}/{len(prompt_tokens)}")
    for i, (o, e) in enumerate(zip(ours_top1, expected)):
        mark = "✓" if o == e else "✗"
        print(f"  pos={i}  ours={o:>6d}  oracle={e:>6d}  {mark}")

    if matches != len(prompt_tokens):
        # Show top-5 of ours for the first mismatch to help diagnose.
        for i, (o, e) in enumerate(zip(ours_top1, expected)):
            if o != e:
                row = logits[i]
                top5 = row.topk(5)
                print(f"\n  ours top-5 at pos {i}:")
                for tok, lv in zip(top5.indices.tolist(), top5.values.tolist()):
                    print(f"    {tok:>6d}  {lv:+.3f}")
                break

    if args.n_gen <= 0:
        return 0 if matches == len(prompt_tokens) else 1

    # --- Decode ---
    print(f"\ngenerating {args.n_gen} tokens...")
    out = list(prompt_tokens)
    for step in range(args.n_gen):
        nxt = int(torch.argmax(logits[-1]).item())
        out.append(nxt)
        start = len(out) - 1
        t0 = time.time()
        logits, kvs, ssms = forward(
            model, [nxt], start_pos=start,
            kv_caches=kvs, ssm_states=ssms,
        )
        print(f"  step {step}: tok={nxt}  {time.time()-t0:.1f}s")

    print(f"\nours full sequence:          {out}")
    print(f"oracle full sequence:        {oracle_all.tolist()}")
    print(f"full match: {out == oracle_all.tolist()}")
    return 0 if out == oracle_all.tolist() else 1


if __name__ == "__main__":
    raise SystemExit(main())
