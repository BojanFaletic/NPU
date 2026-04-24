"""Qwen3.5/3.6 MoE hybrid model — config + tensor loader.

Loads the GGUF file lazily: tensor metadata is indexed on init, but weights
are only dequantized on first access (and cached thereafter). This keeps
resident memory near the quantized-on-disk footprint (~16 GB mmap'd) rather
than the ~70 GB that full bf16 dequantization would cost.

GGUF shape convention: t.shape[0] is the fastest-varying dim (row-major /
"in_features first" for linear weights). Pytorch's Linear.weight convention is
(out, in). We expose weights in GGML order (in, out): `W @ x.T` style matmul
rather than `x @ W.T`. This avoids an unnecessary transpose for every tensor
and matches how llama.cpp operates internally.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    arch: str
    n_layer: int
    n_ctx_train: int
    d_model: int              # embedding_length
    n_head: int               # attention: Q heads
    n_head_kv: int            # attention: KV heads
    head_dim: int             # attention: qk head dim == v head dim
    rms_eps: float
    rope_theta: float
    rope_dim: int             # partial rotary: only first rope_dim of head
    n_expert: int
    n_expert_used: int
    d_expert_ff: int          # expert hidden
    vocab_size: int
    bos_id: int
    eos_id: int
    pad_id: int | None

    # SSM (Gated DeltaNet) hparams — only the SSM blocks use these.
    ssm_d_conv: int = 4       # conv1d kernel
    ssm_d_inner: int = 4096   # total V dim (d_inner); head_v_dim = d_inner / n_v_heads
    ssm_d_state: int = 128    # head_k_dim == head_v_dim for qwen35moe
    ssm_n_k_heads: int = 16   # num K heads (ssm_n_group); V heads broadcast from these
    ssm_n_v_heads: int = 32   # num V heads (ssm_dt_rank); also the "head count" for α/β

    # Layer type. recurrent_layer_arr[i] = ((i+1) % 4 != 0) per llama.cpp for
    # qwen35moe. Cross-checked against tensor presence (attn_q present = attn).
    is_attention: list[bool] = field(default_factory=list)

    @property
    def d_q(self) -> int:
        return self.n_head * self.head_dim

    @property
    def d_kv(self) -> int:
        return self.n_head_kv * self.head_dim


def _get(r: GGUFReader, key: str):
    f = r.fields.get(key)
    if f is None:
        return None
    v = f.contents()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="replace")
    return v


def build_config(r: GGUFReader) -> Config:
    arch = _get(r, "general.architecture")
    if arch is None:
        raise ValueError("missing general.architecture")
    p = f"{arch}."

    n_layer = int(_get(r, p + "block_count"))

    # Discover per-layer type by tensor presence.
    layer_has_attn_q = [False] * n_layer
    for t in r.tensors:
        if ".attn_q.weight" in t.name and t.name.startswith("blk."):
            i = int(t.name.split(".")[1])
            layer_has_attn_q[i] = True
    is_attention = layer_has_attn_q

    # Head counts from actual weight shapes. Metadata head_count is OK here
    # (16) but we cross-check against shapes.
    #
    # Qwen3-Next attention is *gated*: the attn_q projection outputs
    #   [q (n_head * head_dim) | q_gate (n_head * head_dim)] = 2*n_head*head_dim
    # so n_head must be derived from attn_output's input dim (which equals
    # n_head * head_dim post-gate), not from attn_q's output dim.
    head_dim = int(_get(r, p + "attention.key_length"))
    n_head = None
    n_head_kv = None
    for t in r.tensors:
        if t.name.endswith(".attn_output.weight"):
            in_dim = int(t.shape[0])
            n_head = in_dim // head_dim
            break
    for t in r.tensors:
        if t.name.endswith(".attn_k.weight"):
            out_dim = int(t.shape[1])
            n_head_kv = out_dim // head_dim
            break
    if n_head is None or n_head_kv is None:
        raise ValueError("couldn't derive n_head / n_head_kv from shapes")

    return Config(
        arch=arch,
        n_layer=n_layer,
        n_ctx_train=int(_get(r, p + "context_length")),
        d_model=int(_get(r, p + "embedding_length")),
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim=head_dim,
        rms_eps=float(_get(r, p + "attention.layer_norm_rms_epsilon")),
        rope_theta=float(_get(r, p + "rope.freq_base")),
        rope_dim=int(_get(r, p + "rope.dimension_count")),
        n_expert=int(_get(r, p + "expert_count")),
        n_expert_used=int(_get(r, p + "expert_used_count")),
        d_expert_ff=int(_get(r, p + "expert_feed_forward_length")),
        ssm_d_conv=int(_get(r, p + "ssm.conv_kernel") or 4),
        ssm_d_inner=int(_get(r, p + "ssm.inner_size") or 4096),
        ssm_d_state=int(_get(r, p + "ssm.state_size") or 128),
        ssm_n_k_heads=int(_get(r, p + "ssm.group_count") or 16),
        ssm_n_v_heads=int(_get(r, p + "ssm.time_step_rank") or 32),
        vocab_size=int(_get(r, "tokenizer.ggml.tokens") is not None
                       and len(_get(r, "tokenizer.ggml.tokens")) or 0),
        bos_id=int(_get(r, "tokenizer.ggml.bos_token_id") or 0),
        eos_id=int(_get(r, "tokenizer.ggml.eos_token_id") or 0),
        pad_id=_get(r, "tokenizer.ggml.padding_token_id"),
        is_attention=is_attention,
    )


# ---------------------------------------------------------------------------
# Tensor loader
# ---------------------------------------------------------------------------

class TensorStore:
    """Lazy tensor store. `get(name, dtype=np.float32 | np.float16 | "bf16")`
    dequantizes on first access, caches. Pass `keep=False` to skip caching
    (for one-shot large tensors like the embedding matrix).

    Note: bf16 is represented as uint16 in numpy (no native bf16 dtype). Use
    `to_bf16(arr_fp32) -> uint16` and `from_bf16(arr_uint16) -> fp32` for
    conversion. The NPU path consumes the uint16 view directly.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.reader = GGUFReader(str(self.path))
        self.cfg = build_config(self.reader)
        self._by_name = {t.name: t for t in self.reader.tensors}
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def names(self) -> list[str]:
        return list(self._by_name)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def raw(self, name: str):
        t = self._by_name.get(name)
        if t is None:
            raise KeyError(name)
        return t

    def get(self, name: str, dtype: str = "fp32",
            keep: bool = True) -> np.ndarray:
        """Return dequantized tensor in GGML shape order (in, out)."""
        if dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError(dtype)
        key = (name, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        t = self._by_name.get(name)
        if t is None:
            raise KeyError(name)

        if t.tensor_type == GGMLQuantizationType.F32:
            arr = np.asarray(t.data).view(np.float32).reshape(
                tuple(int(x) for x in t.shape))
        else:
            arr = dequantize(t.data, t.tensor_type)
            arr = arr.reshape(tuple(int(x) for x in t.shape))
            arr = arr.astype(np.float32, copy=False)

        if dtype == "fp16":
            arr = arr.astype(np.float16)
        elif dtype == "bf16":
            arr = to_bf16(arr)

        if keep:
            self._cache[key] = arr
        return arr

    def drop(self, name: str, dtype: str = "fp32") -> None:
        self._cache.pop((name, dtype), None)

    def cache_bytes(self) -> int:
        return sum(a.nbytes for a in self._cache.values())


def to_bf16(arr_fp32: np.ndarray) -> np.ndarray:
    """fp32 → bf16 stored as uint16. Round-to-nearest-even via the top 16 bits
    of the fp32 representation, with a rounding bias."""
    if arr_fp32.dtype != np.float32:
        arr_fp32 = arr_fp32.astype(np.float32)
    u32 = arr_fp32.view(np.uint32)
    # round half to even: add the value such that truncation rounds correctly.
    # Standard trick: add 0x7FFF + ((u32 >> 16) & 1), then take high 16 bits.
    rounded = u32 + 0x7FFF + ((u32 >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def from_bf16(arr_u16: np.ndarray) -> np.ndarray:
    """bf16 (stored as uint16) → fp32."""
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    default="qwen/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf")
    ap.add_argument("--sample", type=int, default=3,
                    help="dequantize the first N blocks' tensors as smoke test")
    ap.add_argument("--dtype", default="bf16",
                    choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    t0 = time.time()
    ts = TensorStore(args.path)
    t_open = time.time() - t0
    cfg = ts.cfg

    print(f"== config ==")
    print(f"  arch:         {cfg.arch}")
    print(f"  n_layer:      {cfg.n_layer}  "
          f"({sum(cfg.is_attention)} attn / "
          f"{cfg.n_layer - sum(cfg.is_attention)} ssm)")
    print(f"  attn layers:  {[i for i,a in enumerate(cfg.is_attention) if a]}")
    print(f"  d_model:      {cfg.d_model}")
    print(f"  n_head:       {cfg.n_head} Q / {cfg.n_head_kv} KV  "
          f"(head_dim={cfg.head_dim}, rope_dim={cfg.rope_dim})")
    print(f"  rope_theta:   {cfg.rope_theta:.2e}")
    print(f"  rms_eps:      {cfg.rms_eps:.2e}")
    print(f"  experts:      {cfg.n_expert_used}/{cfg.n_expert} used  "
          f"(ff={cfg.d_expert_ff})")
    print(f"  vocab:        {cfg.vocab_size}  bos={cfg.bos_id} eos={cfg.eos_id}")
    print(f"  ctx_train:    {cfg.n_ctx_train}")
    print(f"  open+scan:    {t_open*1000:.0f} ms  ({len(ts.names())} tensors)")

    # Dequant smoke test: first N blocks, check dtype and shape.
    print(f"\n== dequant smoke test (dtype={args.dtype}) ==")
    total_bytes = 0
    total_time = 0.0
    sampled = 0
    for name in ts.names():
        if not name.startswith("blk."):
            continue
        i = int(name.split(".")[1])
        if i >= args.sample:
            break
        t0 = time.time()
        arr = ts.get(name, dtype=args.dtype)
        dt = time.time() - t0
        total_time += dt
        total_bytes += arr.nbytes
        sampled += 1
        ok = np.isfinite(from_bf16(arr) if args.dtype == "bf16"
                         else arr.astype(np.float32)).all()
        print(f"  {name:<55s}  shape={str(arr.shape):<22s}  "
              f"{arr.dtype!s:<8s}  {arr.nbytes/1e6:7.2f} MB  "
              f"{dt*1000:7.1f}ms  {'OK' if ok else 'BAD'}")

    print(f"\n  {sampled} tensors / {total_bytes/1e9:.2f} GB "
          f"dequanted in {total_time*1000:.0f} ms  "
          f"({total_bytes/1e9/max(total_time,1e-9):.2f} GB/s)")
    print(f"  cache resident: {ts.cache_bytes()/1e9:.2f} GB")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
