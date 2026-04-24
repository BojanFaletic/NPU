"""Print metadata + tensor table summary for a GGUF file."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from gguf import GGUFReader, GGMLQuantizationType


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    default="qwen/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf")
    ap.add_argument("--tensors", action="store_true",
                    help="dump every tensor name, shape, quant")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"missing: {path}", file=sys.stderr)
        return 2

    r = GGUFReader(str(path))

    print(f"== file ==")
    print(f"  path: {path}  ({path.stat().st_size / 1e9:.2f} GB)")

    print("\n== metadata (selected) ==")
    wanted = [
        "general.architecture",
        "general.name",
        "general.basename",
        "general.size_label",
        "general.file_type",
        "general.quantization_version",
        "general.quantized_by",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
    ]
    for name in wanted:
        f = r.fields.get(name)
        if f is None:
            continue
        try:
            val = f.contents()
        except Exception as e:
            val = f"<err: {e}>"
        if isinstance(val, (bytes, bytearray)):
            val = val.decode("utf-8", errors="replace")
        if isinstance(val, str) and len(val) > 120:
            val = val[:117] + "..."
        print(f"  {name}: {val!r}")

    print("\n== arch hyperparams ==")
    arch_field = r.fields.get("general.architecture")
    arch = arch_field.contents() if arch_field else None
    if isinstance(arch, (bytes, bytearray)):
        arch = arch.decode()
    prefix = f"{arch}." if arch else ""
    hparam_keys = [
        f"{prefix}block_count",
        f"{prefix}context_length",
        f"{prefix}embedding_length",
        f"{prefix}feed_forward_length",
        f"{prefix}attention.head_count",
        f"{prefix}attention.head_count_kv",
        f"{prefix}attention.key_length",
        f"{prefix}attention.value_length",
        f"{prefix}attention.layer_norm_rms_epsilon",
        f"{prefix}rope.freq_base",
        f"{prefix}rope.dimension_count",
        f"{prefix}rope.scaling.type",
        f"{prefix}rope.scaling.factor",
        f"{prefix}expert_count",
        f"{prefix}expert_used_count",
        f"{prefix}expert_feed_forward_length",
        f"{prefix}expert_shared_count",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
    ]
    for k in hparam_keys:
        f = r.fields.get(k)
        if f is None:
            continue
        try:
            val = f.contents()
        except Exception as e:
            val = f"<err: {e}>"
        print(f"  {k}: {val!r}")

    print("\n== tensor quant summary ==")
    quant_counter: Counter[str] = Counter()
    quant_bytes: Counter[str] = Counter()
    for t in r.tensors:
        qname = GGMLQuantizationType(t.tensor_type).name
        quant_counter[qname] += 1
        quant_bytes[qname] += int(t.n_bytes)
    total_b = sum(quant_bytes.values())
    for qname, n in quant_counter.most_common():
        b = quant_bytes[qname]
        print(f"  {qname:>12s}: {n:>4d} tensors, {b/1e9:6.2f} GB "
              f"({100*b/total_b:5.1f}%)")

    print(f"\n  total: {len(r.tensors)} tensors, {total_b/1e9:.2f} GB")

    print("\n== per-role quant (for key tensor families) ==")
    role_quants: dict[str, Counter[str]] = {}
    for t in r.tensors:
        parts = t.name.split(".")
        if parts[0] == "blk" and len(parts) >= 3:
            role = ".".join(parts[2:])
        else:
            role = t.name
        role = role.replace(".weight", "").replace(".bias", "_bias")
        qname = GGMLQuantizationType(t.tensor_type).name
        role_quants.setdefault(role, Counter())[qname] += 1
    for role, c in sorted(role_quants.items()):
        parts = ", ".join(f"{q}×{n}" for q, n in c.most_common())
        print(f"  {role:>28s}: {parts}")

    if args.tensors:
        print("\n== all tensors ==")
        for t in r.tensors:
            qname = GGMLQuantizationType(t.tensor_type).name
            shape = tuple(int(x) for x in t.shape)
            print(f"  {qname:>8s}  {shape!s:>22s}  {t.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
