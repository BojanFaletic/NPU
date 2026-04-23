//===- fa_kernel.cc ---------------------------------------*- C++ -*-===//
//
// FlashAttention-2 forward pass (causal) for AIE2p / XDNA 2.
// Matches npu/fa_ref.py block-for-block — that file is the correctness oracle.
//
// --- STATUS: SKELETON ---
// The outer structure (types, sizes, block-update signature, finalise) is
// fixed; the inner bodies (matmul tiles, exp, rowmax, rescale) are TODOs
// flagged inline. See npu/fa_ref.py for the fp32 algorithm to mirror.
//
// --- Tile sizing ---
// SmolLM2-135M: Dh = 64.  AIE2p tile DMEM ≈ 64 KB.
//   Br = 32 (query block rows), Bc = 64 (kv block cols), D = 64 (head dim)
// Memory: 4KB Qi + 8KB Kj + 8KB Vj + 8KB S + 8KB O + 256B (m,l) ≈ 36.5 KB ✓
//
// Matmul shapes hit the native bf16 mac:
//   S = Q[Br,D] @ K^T[D,Bc]    (m,k,n = 32, 64, 64)   fits (4,8,8) mac
//   O += P[Br,Bc] @ V[Bc,D]    (m,k,n = 32, 64, 64)   fits (4,8,8) mac
//
// --- Data flow per dispatch (one layer's attention) ---
// Input : Q [B,H,Tq,D], K [B,H,Tk,D], V [B,H,Tk,D]   all bf16
// Output: O [B,H,Tq,D]                                bf16 (cast from f32)
// For SmolLM2 prefill B=1, H=9 (post-GQA).  Kernel iterates over (h, i-qblock)
// outer-loop, (j-kvblock) inner-loop. Multi-core split is along the H dim
// (9 heads across 4 cores = 3+2+2+2, easy static partition).
//
// --- Online softmax invariant ---
// After processing kv-blocks 0..J-1, for every query row r in Qi:
//     m_r = max over cols seen so far of   S[r, c] / sqrt(D)  (with causal mask)
//     l_r = sum over cols seen so far of   exp(S[r,c] - m_r)
//     O_r = sum over cols seen so far of   exp(S[r,c] - m_r) * V[c, :]
// At end of loop:   out_r = O_r / l_r   (all rows, bf16).
// Guard: if l_r == 0 (row fully masked, can happen at decode edges) → write zero.

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// Compile-time constants (defaults; may be overridden with -D). Kept small and
// divisible by the native mac shape (4,8,8).
#ifndef FA_BR
#define FA_BR 32
#endif
#ifndef FA_BC
#define FA_BC 64
#endif
#ifndef FA_D
#define FA_D  64
#endif

constexpr int Br = FA_BR;
constexpr int Bc = FA_BC;
constexpr int D  = FA_D;

// Native bf16 mac shape on AIE2p.
constexpr int R_MAC = 4;
constexpr int S_MAC = 8;
constexpr int T_MAC = 8;

static_assert(Br % R_MAC == 0, "Br must be multiple of 4");
static_assert(D  % S_MAC == 0, "D must be multiple of 8");
static_assert(Bc % T_MAC == 0, "Bc must be multiple of 8");
static_assert(Bc % S_MAC == 0, "Bc must be multiple of 8 (used as k of 2nd matmul)");

// Softmax vector width (matches softmax.cc).
constexpr int SM_VEC = 32;
static_assert(Bc % SM_VEC == 0, "Bc must be multiple of 32 for vectorised softmax");
constexpr float LOG2E = 1.4453125f;  // matches softmax.cc

// -----------------------------------------------------------------------------
// TODO #1: S = (Q @ K^T) * inv_sqrt_d       [Br, Bc]  f32 accum
// Mirror the inner matmul kernel in aie_kernels/aie2p/mm.cc, but:
//   - B operand is K viewed as [Bc, D], we need (row of Q)·(row of K), i.e.
//     k_j[c, :] · q_i[r, :].  Either: (a) pre-transpose K at DMA level, or
//     (b) use the b-column-major variant of the mac.
//   - Scale the accumulator by inv_sqrt_d before the softmax step (cheap:
//     fused into the mul-reduction path).
// Output: accum<accfloat, Br*Bc>  (but laid out as Br rows for later rowmax).
// -----------------------------------------------------------------------------
template<bool apply_causal>
static inline void matmul_Q_Kt(
    const bfloat16 *restrict Qi,   // [Br, D]
    const bfloat16 *restrict Kj,   // [Bc, D]
    accum<accfloat, Br*Bc> &S,
    float inv_sqrt_d,
    int row_base, int col_base) {
  // TODO: write the (Br × D) · (Bc × D)^T matmul using aie::mmul.
  // Apply causal mask if apply_causal: for each (r, c) with r + row_base < c + col_base,
  // set S[r, c] = -inf (use bfloat16 -inf = 0xFF80 sentinel in the bf16 output,
  // but since we accumulate in f32, just set accum lanes to -INFINITY).
  (void)Qi; (void)Kj; (void)S; (void)inv_sqrt_d; (void)row_base; (void)col_base;
}

// -----------------------------------------------------------------------------
// TODO #2: online softmax update.
// Given S_ij [Br, Bc] f32, running (m, l) f32 and running O [Br, D] f32:
//   m_new = max(m, rowmax(S_ij))                 [Br]
//   alpha = exp2((m - m_new) * log2e)            [Br]   (exp via exp2 trick)
//   P     = exp2((S_ij - m_new) * log2e)         [Br, Bc]
//   l     = alpha * l + rowsum(P)                [Br]
//   O     = alpha * O   (broadcast across D)     [Br, D]
//   (then accumulate += P @ V below)
// m = m_new
// Handle the "m_new = -inf" degenerate row (all-masked): force alpha=0, P=0.
// -----------------------------------------------------------------------------
static inline void online_softmax_update(
    accum<accfloat, Br*Bc> &S,
    float *restrict m,      // [Br] in/out
    float *restrict l,      // [Br] in/out
    float *restrict O,      // [Br, D] in/out (f32 accum)
    vector<bfloat16, Br*Bc> &P_out) {
  // TODO: implement rowmax, exp2, rescale, rowsum, broadcast-rescale of O.
  (void)S; (void)m; (void)l; (void)O; (void)P_out;
}

// -----------------------------------------------------------------------------
// TODO #3: O += P @ V    with P stored as bf16 after softmax
// Standard matmul accumulation into f32. Shapes: [Br, Bc] @ [Bc, D] -> [Br, D]
// -----------------------------------------------------------------------------
static inline void matmul_P_V_accum(
    const vector<bfloat16, Br*Bc> &P,
    const bfloat16 *restrict Vj,
    float *restrict O) {
  // TODO: write the P @ V matmul accumulating into O.
  (void)P; (void)Vj; (void)O;
}

// -----------------------------------------------------------------------------
// TODO #4: finalise — O_out[r, :] = O[r, :] / l[r], cast to bf16.
// Guard: if l[r] == 0, write zeros (row fully masked).
// -----------------------------------------------------------------------------
static inline void finalise(
    const float *restrict O,   // [Br, D]
    const float *restrict l,   // [Br]
    bfloat16 *restrict O_out   // [Br, D]
    ) {
  // TODO: broadcast divide + cast.
  (void)O; (void)l; (void)O_out;
}

// =============================================================================
// Public entry. One call = one (Qi query block) ingestion, all K/V blocks
// streamed in. Caller (IRON orchestration) pushes K/V blocks in order via the
// ObjectFifos; this function reads them via the Kj/Vj pointers one block at
// a time. The caller also signals 'is_first' on the first kv-block so we
// initialise (m=-inf, l=0, O=0) before updating.
//
// For the MVP we bake in the kv-block count (n_kblocks) at compile time via
// -DFA_NKB. Later we can make it runtime.
// =============================================================================
#ifndef FA_NKB
#define FA_NKB 1
#endif

extern "C" {

void fa_forward_bf16(
    const bfloat16 *restrict Q_block,  // [Br, D]
    const bfloat16 *restrict K_stream, // [FA_NKB * Bc * D], concatenated
    const bfloat16 *restrict V_stream, // [FA_NKB * Bc * D], concatenated
    bfloat16 *restrict O_out,          // [Br, D]
    int32_t row_base,                   // absolute row start for this Q block
    int32_t causal                      // 1 if causal, 0 otherwise (typed int for IRON)
) {
  event0();

  // f32 running state
  float m[Br];
  float l[Br];
  float O[Br * D];
  for (int r = 0; r < Br; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }
  for (int i = 0; i < Br * D; ++i) O[i] = 0.0f;

  constexpr float INV_SQRT_D = 1.0f / 8.0f;  // sqrt(64) = 8

  // Scratch for P after softmax (bf16, consumed by second matmul)
  vector<bfloat16, Br*Bc> P;

  for (int j = 0; j < FA_NKB; ++j) {
    const bfloat16 *Kj = K_stream + (size_t)j * Bc * D;
    const bfloat16 *Vj = V_stream + (size_t)j * Bc * D;

    accum<accfloat, Br*Bc> S;
    if (causal) {
      matmul_Q_Kt<true>(Q_block, Kj, S, INV_SQRT_D, row_base, j * Bc);
    } else {
      matmul_Q_Kt<false>(Q_block, Kj, S, INV_SQRT_D, row_base, j * Bc);
    }
    online_softmax_update(S, m, l, O, P);
    matmul_P_V_accum(P, Vj, O);
  }

  finalise(O, l, O_out);

  event1();
}

} // extern "C"

// =============================================================================
// NEXT-SESSION WORK ORDER
// =============================================================================
// 1. Implement matmul_Q_Kt (TODO #1) — cheapest first test: set causal=false,
//    verify S values match Q @ K^T / sqrt(D) bit-for-bit against a CPU oracle.
//    Use pattern from aie_kernels/aie2p/mm.cc.
// 2. Implement online_softmax_update (TODO #2) — this is the heart. The
//    rowmax, exp2, rescale idioms are all in softmax.cc; adapt for per-row.
// 3. Implement matmul_P_V_accum (TODO #3) — same matmul kernel as #1, with
//    accumulate (not zero-init).
// 4. Implement finalise (TODO #4) — broadcast divide + cast.
// 5. Write fa.py (IRON + dispatch wrapper, modeled on npu/linear.py). Build
//    xclbin per (Tq, Tk) shape, each dispatch = one attention call.
// 6. Wire into smollm.py Layer.forward replacing the three CPU ops
//    (Q·K^T, softmax, att·V). Verify top-1 matches HF + bench.
//
// Memory-layout note: Q_block is expected in [Br, D] row-major. K_stream and
// V_stream are expected as [n_kblocks × Bc × D] row-major (Kj at stride
// Bc·D from the previous block). If IRON's dims_to_stream makes row-major
// K tiles easier via b-col-major access, switch matmul_Q_Kt to the
// b-col-major mmul variant — either works, one just saves a transpose.
