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

## Next session — attention on NPU

Current NPU offload only handles the projection matmuls. The attention body
(`Q·K^T`, softmax, `att·V`, output `o_proj`) still runs on CPU:

- `Q·K^T` has shape `[B, H, T, T]` — varies with `T`, doesn't fit the
  fixed-shape xclbin model cleanly.
- Softmax is per-row and numerically sensitive.
- `att·V` is another `[B, H, T, T] · [B, H, T, Dh]`.

`vendor/mlir-aie-src/aie_kernels/aie2p/` ships ready-made AIE2p kernels:
`softmax.cc`, `layer_norm.cc`, `rms_norm.cc`, `rope.cc`, `silu.cc`,
`swiglu.cc`, and `mm_bfp.cc` (block-fp ≈ int8 matmul). **These cover almost
everything we still need** — the softmax kernel is the jump-in point for an
NPU-resident attention.

Suggested plan:
1. Write a **FlashAttention-style kernel** that streams Q-blocks through a
   compute tile and computes `softmax(Q·K^T)·V` with online rescaling, so
   the `T²` intermediate never materialises.
2. Alternative stepping stone: a standalone softmax kernel wrapped like
   `NpuLinear`, used as a sanity test before tackling full FA.
3. Once attention runs on NPU, measure end-to-end: the CPU↔NPU PCIe/DMA
   roundtrip between projections and attention is currently a big cost.
4. After attention — quantization path (`mm_bfp.cc`, int8 weights, 2×
   throughput + half the DMA). This is also the path toward Qwen3-27B.
