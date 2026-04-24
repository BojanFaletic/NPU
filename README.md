# NPU — SmolLM2-135M on AMD XDNA 2

Running a Llama-style LLM on the AMD Ryzen AI NPU (XDNA 2 / Krackan Point),
end to end, with hand-written kernels. The 135M model is the learning vehicle;
the end goal is **`unsloth/Qwen3.6-35B-A3B-GGUF`** — a 35B MoE (≈3B active per
token) running efficiently on the same laptop, with the NPU carrying the
attention + expert matmul load.

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
- **FlashAttention-2 on NPU** (`npu/fa.py`, `npu/fa_kernel.cc`) — one
  dispatch processes one Q block (BR=32 rows) attending to all TK keys.
  Host streams n_kv = TK/BC (BC=32) key/value block pairs; kernel keeps
  running softmax state (row-max, row-sum, output accumulator) in static
  tile memory so the [BR, TK] intermediate never materialises. Causal
  mask supported via a tiny header in the Q buffer (`start_row` +
  `causal` flag). Tested TK up to 512 with causal, max|Δ|≈4.7e-2.
- **SmolLM integration**: `smollm.py --npu` enables FA for prefill (T>1).
  One dispatch per layer; the N = Hq × n_q Q-blocks are split across **4
  compute cores** via memtile-aggregated ObjectFifo.split/join (padded to
  a multiple of 4). Inside the kernel, Q·K^T uses `aie::mmul<4,8,8>` on
  host-pre-tiled Q and K (no per-element reduce_add); softmax is
  vectorised over BC (1 exp2 per row); S·V still uses a scalar-P_row
  vector MAC (mmul there broke top-1 precision across kv-blocks). **Top-1
  matches HF** on tested prompts. End-to-end prefill 4-7× slower than
  CPU, down from 30-180× at session start:

  ```
  T    cpu(ms)  npu(ms)  ratio
   16     47      346    0.135x
   32     54      374    0.143x
   64     67      339    0.198x
  128     99      379    0.262x
  ```

  Session arc: four kernel optimisations (softmax vec → mmul Q·K^T →
  per-layer head batching → 4-core split). Kernel-only time for T=128
  went 38.9 ms → 4.25 ms (9×); full end-to-end went 1400 ms → 379 ms (3.7×).
- **Fused QKV + gate/up projections** — `Layer` pre-concats `[wq;wk;wv]` into
  a single `[960, 576]` weight and `[w_gate; w_up]` into `[3072, 576]`; one
  NPU dispatch replaces three + two per layer, saving 90 launches per
  forward. ~8% win at T=128, flat at shorter T (FA + remaining projections
  dominate there). Output is sliced on the N axis.
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
  fa_kernel.cc      — AIE2p FlashAttention-2 kernel: init / block / finalise
                      entry points, running softmax state in static DMEM,
                      scalar matmuls, aie::exp2-backed scalar exp helper.
  fa.py             — NpuAttention dispatch wrapper + self-test. 2 input
                      FIFOs (Q once, KV pairs streamed), 1 output FIFO.
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

## Roadmap — from SmolLM2-135M to Qwen3.6-35B-A3B

SmolLM2 is now end-to-end on NPU (projections fused + FA-2 multi-core,
top-1 matches HF). Prefill is still 4–7× slower than CPU; decode is CPU.
The ladder from here to `unsloth/Qwen3.6-35B-A3B-GGUF`:

1. **Close the prefill gap on SmolLM2** — fixed per-dispatch overhead
   dominates at short T. Options: merge FA + down-proj into one xclbin,
   batch several layers' launches, or keep all weights device-resident
   so only activations cross PCIe.
2. **Decode path on NPU** — today decode falls back to CPU (T=1 softmax
   + CPU attention). Need an FA-decode kernel (single Q row, TK=full KV)
   and an `NpuLinear` fast path that amortises one xclbin across all
   decode steps.
3. **Quantized matmul** — `vendor/mlir-aie-src/aie_kernels/aie2p` has
   `mm_bfp.cc` (block-fp) plus int8 variants. GGUF's Q4_K/Q5_K/Q8_0
   weight layouts need a dequant/compute kernel on the AIE; this is the
   single biggest win (≈4× DMA, ≈2× compute vs bf16) and the only way a
   35B model fits alongside KV cache in 27 GiB of DRAM.
4. **GGUF loader** — parse unsloth's Qwen3.6 GGUF, stream quantized
   tensors straight into device-side tiled layouts. No bf16 round-trip.
5. **Qwen3 architecture deltas** — GQA with different H_kv ratio,
   RMSNorm placement, SwiGLU, rotary-freq base, **and MoE routing**
   (top-k experts per token out of N). MoE means the per-token expert
   matmuls change every step, which wrecks any "one xclbin per shape"
   cache strategy — need a single templated expert-matmul xclbin plus
   an on-device router.
6. **Qwen3.6-35B-A3B end-to-end** — 3B active params per token keep the
   per-step compute tractable on one NPU; the full 35B weight set stays
   on disk/RAM, streamed by expert. Success metric: interactive decode
   (≥5 tok/s) with top-1 matching the GGUF llama.cpp reference.
