# NPU — SmolLM2-135M on AMD XDNA 2

Running a Llama-style LLM on the AMD Ryzen AI NPU (XDNA 2 / Krackan Point),
end to end, with hand-written kernels. The 135M model is the learning vehicle;
the end goal is Qwen3-27B/32B running efficiently on the same laptop.

## Status

- **CPU reference** (`smollm.py`, `ref_hf.py`) — hand-rolled Llama forward
  matching HuggingFace token-for-token (logits max |Δ| = 3e-5 fp32 rounding).
  KV cache, generates at ~27 tok/s on one core.
- **NPU toolchain** — `amdxdna` kernel driver (in-tree on kernel 6.17) +
  XRT 2.23 + `amdxdna` plugin shim (built from `amd/xdna-driver`) +
  `mlir-aie` 1.3.1 wheel + `llvm-aie` (Peano) nightly. Dispatch via `pyxrt`
  directly — the `aie.xrt` Python binding has a hard `RyzenAI-Phoenix`-only
  assert, so it aborts on Krackan.
- **First kernels** — int32 passthrough (`npu/hello.py`, 0 mismatches) and
  bf16 matmul (`npu/matmul.py`, single-core ~70 GFLOPS, correctness to 4e-5).
- **Multi-core matmul** (`npu/matmul_mc.py`) — 4 columns: ~1070 GFLOPS,
  8 columns: ~1060 GFLOPS on Krackan's 6×8 topology. ~3× torch CPU fp32 BLAS,
  ~5% of chip theoretical peak (~25 TFLOPS bf16).
- **End-to-end forward pass on NPU** — `smollm.py --npu` routes every
  Q/K/V/O/gate/up/down projection through `NpuLinear`. All 30 layers × 7 ops
  dispatched via 4 cached xclbins (K=576/N=576, K=576/N=192, K=576/N=1536,
  K=1536/N=576). **Top-1 token matches HF exactly** on test prompts.
- **NPU softmax** (`npu/softmax.py`) — per-row bf16 softmax backed by the
  stock `aie_kernels/aie2p/softmax.cc` via a thin wrapper that bakes the row
  length in at compile time. One xclbin per `(rows, L_padded)`. Padded to
  multiples of 32 (SM_VEC_LEN) with `-inf`. Enabled during **prefill only**
  (T>1); decode keeps CPU softmax to avoid minting a new xclbin per step.
  Top-1 token still matches HF exactly end-to-end.
- **NPU softmax bench** (`npu/bench_softmax.py`, `bench_smollm.py`) —
  standalone NPU softmax beats CPU above T=256 (~1.5–1.8×), but end-to-end
  it's a regression at every prefill length: +2 ms at T=16 → +866 ms at
  T=1024. Standalone→in-context is a 4–8× blowup, most likely from xclbin
  context-switching between `NpuLinear` and `NpuSoftmax` at every layer. The
  fix is FA-style fusion (softmax inside the matmul xclbin), not more softmax
  tuning.
- **Fused attention on NPU** (`npu/fa.py`, `npu/fa_kernel.cc`) — single
  compute tile runs `softmax(Q·K^T / √D)·V` in one dispatch, Q/K/V packed
  into one input buffer (compute tiles have a 2 input DMA channel limit on
  AIE2p). Scalar matmuls in/out of the stock per-row `softmax_simple_bf16`.
  Working and tested at TQ=TK=32, D=64 — 10/10 trials match torch attention
  to bf16 precision (max|Δ|≈3-5e-2, mean|Δ|≈5e-3). Bigger shapes need
  streaming (the 64×64 ObjectFifo doesn't fit in tile DMEM) — that's the
  next step toward true FA with online rescaling. `npu/fa_ref.py` is the
  Python block-streaming reference (matches torch to fp32 rounding).
- **Benchmarks** — at short prefill lengths NPU is still **slower** than CPU
  (~0.2× at L=16, ~0.9× at L=2048), fixed per-op dispatch overhead dominates.
  Useful prefill win starts needing less driver-Python overhead per op.

## Layout

```
smollm.py           — hand-rolled Llama forward pass, with --npu flag
ref_hf.py           — HuggingFace oracle generator
bench_smollm.py     — full-model forward benchmark, CPU vs NPU, across L
npu/
  hello.py          — int32 passthrough on NPU (toolchain smoke test)
  matmul.py         — single-core bf16 matmul, one shape at a time
  matmul_mc.py      — multi-core (n_aie_cols) matmul using whole_array example
  bench_matmul.py   — throughput sweep (tile sizes, dims)
  linear.py         — NpuLinear: torch-callable bf16 linear layer with
                      xclbin cache and persistent device buffers
  softmax_kernel.cc — thin wrapper around aie_kernels/aie2p/softmax.cc that
                      bakes SM_LEN at compile time
  softmax.py        — NpuSoftmax: per-row bf16 softmax + IRON program +
                      standalone self-test
  bench_softmax.py  — standalone softmax throughput bench (NPU vs CPU)
  fa_ref.py         — FlashAttention-2 forward-pass reference in Python
                      (block streaming + online softmax); bit-matches torch
                      attention to fp32 rounding
  fa_kernel.cc      — AIE2p fused-attention kernel: scalar Q·K^T, stock
                      per-row softmax, scalar P·V. One dispatch = one head
                      of attention.
  fa.py             — NpuAttention dispatch wrapper + self-test
  verify_layer.py   — compares NPU Q-projection output vs CPU inside one layer
  build/            — generated xclbins and object files (gitignored)
vendor/             — gitignored
  xdna-driver/      — AMD source (amd/xdna-driver + XRT submodule) — for build only
  mlir-aie-src/     — sparse checkout at tag v1.3.1 (matches the mlir-aie wheel)
```

## Hardware & software baseline

- Laptop: ThinkPad, **AMD Ryzen AI 7 PRO 350** (Krackan Point, Zen 5/5c),
  Radeon 860M iGPU, 27 GiB system RAM.
- NPU: **XDNA 2 / AIE2p**, PCI `1022:17f0` rev 20, `/dev/accel/accel0`,
  topology 6×8, firmware `NPU Firmware Version : 1.1.2.64`.
- Ubuntu 24.04 LTS, kernel 6.17.0-1017-oem (in-tree `amdxdna` driver).
- Python 3.12 via `uv` (`.venv/` checked into uv workspace, not git).

## Running

Prereqs (one-time):

```sh
# Install system build deps for XRT
cd vendor/xdna-driver && sudo ./tools/amdxdna_deps.sh

# Build and install XRT + NPU plugin (produces .debs, requires sudo to install)
cd vendor/xdna-driver/xrt/build && ./build.sh -npu -opt
sudo apt reinstall ./Release/xrt_*-base.deb ./Release/xrt_*-base-dev.deb ./Release/xrt_*-npu.deb
cd ../../build && ./build.sh -release
sudo apt reinstall ./Release/xrt_plugin*-amdxdna.deb

# Python env (all wheels pinned in pyproject.toml via uv)
uv sync
```

Runtime: `source /opt/xilinx/xrt/setup.sh` then run anything.

```sh
# CPU reference (matches HF)
uv run python smollm.py --check --max-new-tokens 30

# End-to-end NPU forward (first call compiles 4 xclbins, ~25 s total, then cached)
source /opt/xilinx/xrt/setup.sh
uv run python smollm.py --npu --check --max-new-tokens 0

# Multi-core matmul micro-benchmark
uv run python npu/matmul_mc.py -M 512 -K 512 -N 512 -m 32 -k 64 -n 32 --cols 4
```

## Key non-obvious gotchas

- `aie.xrt.XCLBin` has a hard-coded `RyzenAI-Phoenix` assertion — unusable on
  Krackan/Strix. Use `pyxrt` from `/opt/xilinx/xrt/python` instead.
- IRON programs with auto-placed tiles (`AnyShimTile`) need a `SequentialPlacer`
  in `resolve_program(SequentialPlacer())`; otherwise resolution crashes.
- `vendor/mlir-aie-src` is pinned to tag `v1.3.1` to match the IRON API in the
  wheel (newer `main` uses `tile=` kwargs that the wheel's wrapper rejects).
- Firmware dir `amdnpu/17f0_20/` is missing from Ubuntu's `linux-firmware`
  package — driver silently falls back to rev 11 and it works fine.
- `whole_array_iron` tiling requires `M % (m · n_aie_rows · tb_n_rows) == 0`
  i.e. `M % (m·8) == 0` for the default tb layout. `NpuLinear.Plan` pads
  accordingly (minimum M_pad = 256 with `m=32`).

## Next session — scale FA kernel up

Single-block fused attention works at TQ=TK=32, D=64. To use this on SmolLM2
prefill (needs T up to hundreds of tokens), the kernel has to stream K/V
blocks through a fixed-size Q block with online softmax rescaling — the
actual FlashAttention algorithm. The Python reference for that already
exists (`npu/fa_ref.py`, block streaming + online softmax, bit-matches
torch to fp32 rounding).

Work order:
1. Convert the C++ kernel from "one block, T² on-chip" to "stream K/V blocks
   with online softmax stats (m, l) and output rescaling". This is where the
   kernel graduates from "fused attention" to "FA".
2. Add causal masking in the Q·K^T step (positions `c > r + start_pos` set
   to -inf pre-softmax).
3. IRON orchestration: loop K/V blocks via a second ObjectFifo streaming
   through the compute tile, so the tile only holds one (Qi, Kj, Vj) at a
   time — memory budget drops from O(T²) to O(Br·D + Bc·D).
4. Multi-core split: SmolLM2 has Hq=9 heads, trivial to partition across 4
   compute cores (3+2+2+2). Needed for end-to-end throughput to beat CPU.
5. Wire into `smollm.py` attention body, verify top-1 matches HF + bench.

After FA works end-to-end: quantization (`mm_bfp.cc`, int8 weights, 2×
throughput + half the DMA), path toward Qwen3-27B.
