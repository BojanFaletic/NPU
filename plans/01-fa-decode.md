# Plan 01 — NPU FA-decode via bucketed Tk

**Status**: ready for implementation
**Estimated scope**: ~100 LOC net; no new kernel; no new xclbin compile script
**Delegation**: Codex via @bojan

## TL;DR

Route T=1 (decode) attention through the existing NPU FlashAttention-2 kernel instead of falling back to CPU. To avoid minting a new xclbin every decode step as KV grows, pad Tk up to a log-spaced bucket set. Gate on `bench_chatbot.py` correctness.

## Why this task (honest framing)

At decode, CPU attention is already essentially free (~2 ms/forward across all 30 layers). Routing attention to NPU **adds** roughly 30 ms/step (one FA dispatch per layer × 1 ms dispatch floor). **This plan therefore expects a ~10% decode-step regression, not an improvement.**

We do it anyway because:
- It's infrastructure. The next plan (layer fusion) needs an attention kernel that can run at T=1.
- Log-spaced Tk bucketing is reusable for long-prefill and for Qwen later.
- It pushes the model closer to "fully on NPU, no CPU path" — the stated project goal.

If the 10% decode regression is not acceptable, **pause and ask before coding.** The real decode unlock is whole-layer fusion (a later plan), not this one.

## Repo context (cold-read)

- Working dir: `/home/bojan/fun/NPU`. SmolLM2-135M hand-written Llama forward in `smollm.py`. NPU kernels in `npu/`.
- NPU path: `smollm.py --npu`. End-to-end forward on XDNA 2, correctness matches HF top-1.
- Today `Layer.forward` gates the FA branch on `T > 1`. At T=1 the attention uses `torch.matmul` + `F.softmax` on CPU, while NpuLinear projections (wqkv, wo, w_gate_up, w_down) still dispatch to NPU.
- FA kernel: `npu/fa_kernel.cc` (AIE2p, BR=32, BC=32, D=64 hardcoded as `-DFA_*` macros). Maintains running softmax state (`g_block_idx`, `g_m`, `g_l`, `g_O`) in static tile memory across the per-block calls within one dispatch. `attn_finalise` resets `g_block_idx=0`. Do not modify this file.
- FA host wrapper: `npu/fa.py` (`NpuAttention.run_batch`). Pads batch of Q-blocks to a multiple of `n_cores=4` with dummy blocks whose `start_row=-1_000_000` so causal masks them out. Keyed xclbins on `(n_kv, n_q_total)`.
- Smollm glue: `smollm.py:Layer._fa_attention` flattens `[B, Hq, T, D]` into a batched N stack and calls `NpuAttention.run_batch`.
- Bench harness: `bench_chatbot.py` runs log-spaced context lengths, with a top-1 HF match as the hard correctness gate. Non-zero exit on mismatch.
- Profiler: `npu/profiler.py` + `PROF.enable()`. Use `--profile` flag on the bench.

## Correctness + success gate

The only truth:

```bash
source /opt/xilinx/xrt/setup.sh
uv run python bench_chatbot.py --lengths 16,64,128,256,512 --iters 3 --warmup 2 --decode-n 16
```

Pass criteria:
1. **Top-1 matches HF at every length** (bench exits 0).
2. **Decode step latency does not regress by more than 15%** vs baseline table below.
3. **Sustained decode tok/s does not regress by more than 15%**.
4. **Prefill does not regress at all** (this change adds bucketing but doesn't change the prefill path's resolved Tk_pad for any bench length — L=16/64/128/256/512 all happen to already be buckets).

Baseline to compare against (current master, NPU path):

| L   | NPU prefill ms | NPU step ms | NPU tok/s |
|-----|---------------:|------------:|----------:|
| 16  | 297            | 253         | 4.0       |
| 64  | 424            | 303         | 3.2       |
| 128 | 509            | 319         | 3.2       |
| 256 | 592            | 248         | 3.2       |
| 512 | 1500           | 249         | 3.9       |

## Design

### 1. Bucket set

Define at module scope in `smollm.py`:

```python
# log-spaced Tk buckets covering [1, max_pos=8192]. All multiples of BC=32 so
# the FA kernel's n_kv = Tk_pad/BC is integral. Each distinct bucket mints one
# xclbin per n_q_total value (i.e. one per {prefill-L, decode} pair). 9 buckets
# × 2 n_q shapes = 18 xclbins max per session, cached to disk.
BUCKETS_TK = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
```

Helper:
```python
def _bucket_tk(tk: int) -> int:
    for b in BUCKETS_TK:
        if b >= tk:
            return b
    raise ValueError(f"Tk={tk} exceeds max bucket {BUCKETS_TK[-1]}")
```

### 2. Drop the T>1 gate

In `Layer.forward`, this line (current approx 166):

```python
if self.npu is not None and "attention" in self.npu and T > 1:
```

becomes

```python
if self.npu is not None and "attention" in self.npu:
```

The stale comment above it (`# Prefill (T>1) can route the whole attention body...`) needs a one-line rewrite describing the new behaviour. Keep the CPU attention branch intact — it's still used when `enable_npu(attention=False)`.

### 3. Swap Tk_pad for bucket

In `Layer._fa_attention`, replace:

```python
Tk_pad = ((Tk + BC - 1) // BC) * BC
```

with:

```python
Tk_pad = _bucket_tk(Tk)
```

Nothing else in `_fa_attention` needs to change. The existing zero-pad branch for K/V already runs when `Tk_pad > Tk`, and the in-kernel causal mask already zeros out cols at positions `> row_pos` — padded cols live at positions `≥ Tk_real`, which is always `≥ row_pos + 1` at decode (row_pos = start_pos = current token index).

### 4. Nothing else

Do not touch:
- `npu/fa.py` (host wrapper already batches padding blocks correctly)
- `npu/fa_kernel.cc` (kernel already handles bucketed-Tk input correctly because it operates block by block)
- `npu/linear.py` (see pitfall below on share_N_pad)

## Steps for Codex

1. Read `smollm.py`, `bench_chatbot.py`, `npu/fa.py`, `npu/fa_kernel.cc` (skim). Read this plan end-to-end.

2. Make the three edits above (add `BUCKETS_TK` + helper, drop `T > 1` gate, swap Tk_pad). Keep the patch tight.

3. Run the **fast** correctness check first:
   ```
   source /opt/xilinx/xrt/setup.sh
   uv run python bench_chatbot.py --lengths 16,64 --iters 1 --warmup 1 --decode-n 4
   ```
   Expect: both lengths pass top-1. First decode step at each length compiles a new xclbin (`fa_32x32x64_n{1,2}_q12_c4`), 5–30s each. Second run is cached.

4. If correctness fails, stop and diagnose. Common causes and checks:
   - `max|Δ|` jumped a lot (>3× baseline): kernel state isn't resetting between dispatches. Confirm `attn_finalise` still sets `g_block_idx=0` in `fa_kernel.cc` (it does on master; don't change). Check whether `NpuAttention.run_batch` is being called with correct `start_rows` for decode: expect `start_rows == [start_pos] * (B*Hq)` when T=1, i.e. all 9 heads share the same start_row = current token position.
   - `max|Δ|` same as baseline but top-1 flipped: precision-adjacent. Re-run a second time (bf16 bench is slightly non-deterministic in kernel reduction order).
   - Shape assertion in `run_batch` (e.g. `TK % BC == 0`): check Tk_pad is picked from the bucket set, not some other value.

5. Run the **full** sweep:
   ```
   uv run python bench_chatbot.py --lengths 16,64,128,256,512 --iters 3 --warmup 2 --decode-n 16
   ```
   First run will compile 5 new decode xclbins (one per L's bucket). Each ~10-30s. Subsequent runs cached.

6. Capture before/after numbers by re-running with and without the change (git stash is fine for A/B). Stash, bench, unstash, bench again. Keep both tables.

7. Write the commit message in the repo's style (see `git log --oneline` for examples — short imperative header followed by a terse body). Include the before/after table.

8. Commit on branch `master` (project convention — no feature branches). Do **not** push.

## Pitfalls (learned the hard way)

- **Do not "save cycles" by switching the residual stream to bf16.** Tried and reverted: top-1 flips at L=64 because fp32→bf16 truncation at each NpuLinear output compounds across 30 layers. Stays fp32.
- **Do not force a shared N_pad on NpuLinear** (`share_N_pad=3072`). Tried: wo pads 576→3072 and inflates compute 5.3×, costing more than the saved xclbin context switches. The `Plan.force_N_pad` infrastructure exists in `npu/linear.py` but is dormant.
- **Expect a small correctness drift**: `max|Δ|` is already ~2.75e-1 at L=16 and ~1.26e-1 at L=64 on master. Top-1 still matches. Padded-KV-cols softmax behaviour contributes a touch more; accept up to 3× on `max|Δ|` as long as top-1 matches.
- **The kernel's `g_causal` flag comes from Q buffer header byte 1.** `_fa_attention` passes `causal=True` today and this plan keeps that. At decode with T=1 causal=True correctly masks cols at position > current token. Don't skip the flag.
- **Bench pre-warming**: `bench_chatbot.py` already has a `pre-warming NPU xclbins:` loop that runs one forward per L before timing. That covers compile cost for prefill xclbins. For **decode** xclbins, the pre-warm calls `model_npu.forward(ids)` at length L — that's prefill, not decode. The first decode step inside the timed bench will still pay compile cost on first run. This is fine if bench is run twice (second run is cached). Consider adding a decode pre-warm if you want single-run clean numbers, but don't sink >15 min on it.

## Out of scope

- No new kernel. BR stays 32. Do not branch the fa_kernel for a BR=4 variant here.
- No fusion with wo or gate_up. Next plan.
- No small-M matvec kernel for linears. Next plan.
- No updates to the README until after the commit lands — don't touch README in this diff.

## Definition of done

- Diff is ≤ ~50 LOC in `smollm.py`, zero in other files.
- `bench_chatbot.py --lengths 16,64,128,256,512` exits 0 with all top-1 matches.
- Decode step time within +15% at every L.
- Prefill time within +5% at every L.
- Commit message includes before/after numbers from the full sweep and a one-line summary.
- Ask @bojan before pushing.
