"""Smoke-test gguf.quants.dequantize on one tensor per quant type in the model.

For each of {F32, Q8_0, IQ3_XXS, IQ4_XS, Q6_K, Q5_K} that appears in the file,
pick a representative tensor, dequantize, and report shape + stats. Validates
that dequantize doesn't crash, doesn't produce NaN/Inf, and yields plausible
values.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    default="qwen/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf")
    args = ap.parse_args()

    r = GGUFReader(args.path)

    # Pick the first tensor we see per quant type.
    picks: dict[str, object] = {}
    for t in r.tensors:
        qname = GGMLQuantizationType(t.tensor_type).name
        if qname not in picks:
            picks[qname] = t

    print(f"found quant types: {list(picks)}")
    print()

    bad = 0
    for qname, t in picks.items():
        shape = tuple(int(x) for x in t.shape)
        nelem = int(np.prod(shape))
        t0 = time.time()
        try:
            out = dequantize(t.data, t.tensor_type)
        except Exception as e:
            print(f"  {qname:>10s}  {t.name:<55s}  FAIL: {e}")
            bad += 1
            continue
        dt = time.time() - t0

        out_flat = out.reshape(-1)[:nelem]
        nfin = int(np.isfinite(out_flat).sum())
        nz = int((out_flat == 0).sum())
        mn = float(out_flat.min())
        mx = float(out_flat.max())
        mean = float(out_flat.mean())
        std = float(out_flat.std())

        status = "OK" if nfin == nelem else f"WARN({nelem-nfin} nonfinite)"
        print(f"  {qname:>10s}  {t.name:<55s}  shape={shape!s:<18s}  "
              f"n={nelem:>9d}  {dt*1000:6.1f}ms  "
              f"range=[{mn:+8.4f},{mx:+8.4f}]  mean={mean:+7.4f}  "
              f"std={std:6.4f}  zeros={100*nz/nelem:4.1f}%  {status}")
        if nfin != nelem:
            bad += 1

    print()
    if bad:
        print(f"FAIL: {bad} tensor(s) had issues")
        return 1
    print("all dequants OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
