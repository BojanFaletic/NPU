"""IQ3_XXS reference dequant — block-streaming, mirrors what the AIE kernel
will see.

GGML IQ3_XXS layout: each 256-element block is 98 bytes
    bytes[0:2]   : d (fp16, the per-block scale)
    bytes[2:66]  : qs[64], one byte per grid index (256-entry × 4-lane LUT)
    bytes[66:98] : 8 × uint32 scale/sign words. Each word covers a 32-element
                   sub-block: bits[31:28] = scale (added to 0.5, multiplied by
                   d/2), bits[27:0] = 4 × 7-bit sign indices into a 128-entry
                   ksigns LUT (each entry = 8 sign bits, 1 per output lane).

A sub-block is laid out as 4 groups of 8 lanes; each group consumes 2 grid
indices (4 lanes each → 8) and 1 sign byte. So qs covers 8 sub-blocks × 4
groups × 2 = 64 indices total, and ksigns covers 8 sub-blocks × 4 groups = 32
sign bytes.

This file gives:
  - GRID  : np.ndarray[(256, 4), fp32]  — the abs-value table after grid_map.
  - KSIGNS: np.ndarray[(128, 8), int8]  — sign bits unpacked, +1 / -1.
  - dequant_block(block_98_bytes) -> fp32[256]   (scalar Python per-block,
    mirrors how the AIE will compute one block).
  - dequant_rows(rows_uint8) -> fp32[..., 256*n_blocks] (vectorized numpy,
    used as the validation oracle).

Both must return identical results to gguf-py's IQ3_XXS.dequantize_blocks.
"""
from __future__ import annotations

import numpy as np

from gguf import GGMLQuantizationType
from gguf.quants import IQ2_XXS, IQ3_XXS as _GGUF_IQ3


# ---------------------------------------------------------------------------
# Constant tables (built once; 4 KB grid + 1 KB ksigns).
# ---------------------------------------------------------------------------

def _build_grid() -> np.ndarray:
    """Return the (256, 4) fp32 grid table referenced by qs[i] (0..255)."""
    _GGUF_IQ3.init_grid()
    g = _GGUF_IQ3.grid                              # (1, 1, 256, 4) fp32
    return np.ascontiguousarray(g.reshape(256, 4))


def _build_ksigns() -> np.ndarray:
    """Return the (128, 8) int8 sign table: ksigns[i, lane] in {+1, -1}."""
    raw = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8)   # (128,)
    bits = (raw[:, None] >> np.arange(8, dtype=np.uint8)[None, :]) & 1
    return np.where(bits == 0, np.int8(1), np.int8(-1))


GRID: np.ndarray = _build_grid()                        # (256, 4) fp32
KSIGNS: np.ndarray = _build_ksigns()                    # (128, 8) int8

assert GRID.shape == (256, 4) and GRID.dtype == np.float32
assert KSIGNS.shape == (128, 8) and KSIGNS.dtype == np.int8


# ---------------------------------------------------------------------------
# Per-block scalar dequant — mirrors what the AIE kernel will do.
# ---------------------------------------------------------------------------

def dequant_block(block_bytes: bytes | np.ndarray) -> np.ndarray:
    """Dequantize a single 98-byte IQ3_XXS block to fp32[256]."""
    if isinstance(block_bytes, (bytes, bytearray)):
        block = np.frombuffer(bytes(block_bytes), dtype=np.uint8)
    else:
        block = np.ascontiguousarray(block_bytes).view(np.uint8).reshape(-1)
    if block.shape != (98,):
        raise ValueError(f"expected 98-byte IQ3_XXS block, got {block.shape}")

    d = float(block[:2].view(np.float16)[0])
    qs = block[2:66]                                  # (64,) uint8
    scales = block[66:98].view(np.uint32)             # (8,) uint32

    out = np.empty(256, dtype=np.float32)
    for sb in range(8):
        sw = int(scales[sb])
        db = d * (0.5 + (sw >> 28)) * 0.5
        sign_idx = [(sw >> (g * 7)) & 0x7F for g in range(4)]
        for g in range(4):
            sb_signs = KSIGNS[sign_idx[g]]            # (8,) int8
            ga = GRID[qs[sb * 8 + g * 2]]             # (4,) fp32
            gb = GRID[qs[sb * 8 + g * 2 + 1]]         # (4,) fp32
            lanes = np.concatenate([ga, gb])          # (8,)
            out[sb * 32 + g * 8: sb * 32 + g * 8 + 8] = (
                db * sb_signs.astype(np.float32) * lanes
            )
    return out


# ---------------------------------------------------------------------------
# Vectorized bulk dequant — used as oracle.
# ---------------------------------------------------------------------------

def dequant_rows(blocks_u8: np.ndarray) -> np.ndarray:
    """blocks_u8: (..., n_blocks, 98) uint8 → fp32 (..., n_blocks*256)."""
    blocks_u8 = np.ascontiguousarray(blocks_u8)
    if blocks_u8.dtype != np.uint8 or blocks_u8.shape[-1] != 98:
        raise ValueError(f"need (..., n, 98) uint8, got {blocks_u8.shape}/{blocks_u8.dtype}")
    flat = blocks_u8.reshape(-1, 98)

    d = flat[:, :2].view(np.float16).astype(np.float32).reshape(-1, 1, 1)
    qs = flat[:, 2:66]                                # (n, 64)
    scales = flat[:, 66:98].view(np.uint32)           # (n, 8)

    db = (d * (0.5 + (scales >> 28).astype(np.float32))[..., None] * 0.5
          ).reshape(-1, 8, 1, 1)                       # (n, 8, 1, 1)

    sign_idx = (scales[..., None] >>
                np.array([0, 7, 14, 21], dtype=np.uint32)) & 0x7F   # (n, 8, 4)
    sign_lanes = KSIGNS[sign_idx].astype(np.float32)                # (n, 8, 4, 8)

    grid_lookups = GRID[qs.reshape(-1, 8, 4, 2)]      # (n, 8, 4, 2, 4) fp32
    lanes = grid_lookups.reshape(-1, 8, 4, 8)

    out = (db * sign_lanes * lanes).reshape(*blocks_u8.shape[:-1], 256)
    return out.reshape(*blocks_u8.shape[:-2], blocks_u8.shape[-2] * 256)


# ---------------------------------------------------------------------------
# Self-test against gguf-py
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Pull a real IQ3_XXS expert tensor and verify against gguf-py."""
    from pathlib import Path
    from gguf import GGUFReader

    path = Path(__file__).resolve().parent.parent / "qwen" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
    r = GGUFReader(str(path))

    # Find any IQ3_XXS tensor.
    t = next(t for t in r.tensors
             if t.tensor_type == GGMLQuantizationType.IQ3_XXS)
    print(f"using {t.name}  shape={tuple(int(s) for s in t.shape)}")

    # Pick one slice (first expert if 3D, or first row if 2D).
    if len(t.shape) == 3:
        chunk = np.ascontiguousarray(t.data[0]).view(np.uint8)
    else:
        chunk = np.ascontiguousarray(t.data).view(np.uint8)
    n_blocks = chunk.size // 98
    chunk = chunk[: n_blocks * 98].reshape(n_blocks, 98)

    # Vectorized
    dv = dequant_rows(chunk)
    # gguf-py oracle
    dr = _GGUF_IQ3.dequantize_blocks(chunk).reshape(-1)
    diff_v = np.abs(dv - dr).max()
    print(f"dequant_rows  max|Δ|={diff_v:.3e}")
    assert diff_v == 0.0, "vectorized dequant must bit-match gguf-py"

    # Scalar block (sanity-check first 4 blocks against same oracle).
    for bi in range(4):
        ds = dequant_block(chunk[bi].tobytes())
        diff_s = np.abs(ds - dr[bi * 256: (bi + 1) * 256]).max()
        assert diff_s == 0.0, f"block {bi}: scalar {diff_s}"
    print(f"dequant_block first 4 blocks bit-match gguf-py")


if __name__ == "__main__":
    _self_test()
