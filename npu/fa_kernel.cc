// FlashAttention-2 forward (optional causal mask), streaming K/V blocks.
//
// Each dispatch processes one Q block (BR rows) attending to all TK keys. The
// host streams n_kv = TK/BC key/value block pairs through the compute tile.
// Running softmax state (row-max, row-sum, output accumulator) lives in static
// tile memory across the per-block calls so we never materialise [BR, TK].
//
// Start-row and causal flag ride in the first two bf16 lanes of the Q buffer:
//   Q_with_header[0] = float(start_row)        — absolute position of Q row 0
//   Q_with_header[1] = float(causal ? 1 : 0)
//   Q_with_header[HEADER_BF16 ..] = actual Q data [BR, DH] row-major
// This keeps the compute tile at its 2 input DMA channel limit (Q + KV).
//
// There is no explicit attn_init entry: the first block of a dispatch is
// identified by a static counter (g_block_idx == 0) and initialises state
// from the header in line. attn_finalise resets the counter for the next
// dispatch.

#define NOCPP
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef FA_BR
#define FA_BR 32
#endif
#ifndef FA_BC
#define FA_BC 32
#endif
#ifndef FA_D
#define FA_D 64
#endif

constexpr int BR = FA_BR;
constexpr int BC = FA_BC;
constexpr int DH = FA_D;

static_assert(DH == 64, "inv_sqrt_d hard-coded for DH=64");
constexpr float INV_SQRT_D = 0.125f;
constexpr float LOG2E      = 1.4453125f;

// Header occupies the first 32 bf16 lanes (64 bytes) of the Q buffer so the
// following Q data stays 64-byte-aligned for the vector iterators.
constexpr int HEADER_BF16 = 32;

// AIE2p math.h ships no scalar expf; aie::exp2 is the only exp primitive.
// Broadcast into a 32-lane vector, exp2, extract lane 0. Wasteful but the
// block loop calls it BC+1 times per row, dwarfed by the scalar matmuls.
static inline float exp_scalar(float x) {
  aie::vector<float, 32> v = aie::broadcast<float, 32>(x * LOG2E);
  aie::vector<bfloat16, 32> e = aie::exp2<bfloat16>(v);
  return (float)e.get(0);
}

// Persistent per-dispatch running state. g_O is bf16 (not fp32) to stay inside
// the compute tile's .bss budget; each block update casts to fp32 locally,
// does the rescale+MAC, and casts back — precision loss is per-update bf16,
// which the end-to-end self-test accepts at ~1e-2. alignas because g_m and
// g_O are loaded as BR-lane fp32 / BR×DH bf16 vectors in the hot loop.
static int   g_block_idx = 0;
static int   g_start_row = 0;
static int   g_causal    = 0;
alignas(64) static float g_m[BR];
alignas(64) static float g_l[BR];
alignas(64) static bfloat16 g_O[BR * DH];

extern "C" {

void attn_block(bfloat16 *restrict Q_with_header, bfloat16 *restrict KV) {
  // On the first block of a dispatch, pull start_row + causal flag from the
  // header and zero the running state. bf16 can represent integers up to
  // 2^8 * some rounding — SmolLM2 max_pos=8192, well within exact range.
  if (g_block_idx == 0) {
    g_start_row = (int)(float)Q_with_header[0];
    g_causal    = (int)(float)Q_with_header[1];
    for (int r = 0; r < BR; ++r) {
      g_m[r] = -INFINITY;
      g_l[r] = 0.0f;
    }
    for (int i = 0; i < BR * DH; ++i) g_O[i] = (bfloat16)0.0f;
  }

  bfloat16 *Q = Q_with_header + HEADER_BF16;
  bfloat16 *K = KV;
  bfloat16 *V = KV + BC * DH;
  const int start_col = g_block_idx * BC;

  // In causal prefill, whole KV blocks can sit strictly beyond every row in
  // this Q block. They would be masked to -inf, so skip all QK/softmax/PV math.
  if (g_causal && start_col > g_start_row + BR - 1) {
    g_block_idx++;
    return;
  }

  // S = Q · K^T · inv_sqrt_d. The host pre-tiles Q into [BR/r][D/s][r][s] and
  // K into [D/s][BC/t][s][t] with r=4, s=8, t=8 so each tile is a contiguous
  // 64-byte (32 bf16) vector. aie::mmul<r,s,t> produces a 4×8 fp32 output
  // tile per call, accumulating over D/s inner steps, and doesn't need a
  // per-element reduce_add (the accumulator IS the output tile).
  constexpr int R_MAC = 4;
  constexpr int S_MAC = 8;
  constexpr int T_MAC = 8;
  constexpr int MR = BR / R_MAC;     // 8 output row-tiles
  constexpr int MT = BC / T_MAC;     // 4 output col-tiles
  constexpr int MK = DH / S_MAC;     // 8 accumulation steps
  static_assert(BR % R_MAC == 0 && BC % T_MAC == 0 && DH % S_MAC == 0,
                "Tile sizes must divide BR/BC/DH");

  using MMUL = aie::mmul<R_MAC, S_MAC, T_MAC, bfloat16, bfloat16, accauto>;

  alignas(64) float S_tiled[BR * BC];  // [MR][MT][r][t] row-major tile order
  for (int i = 0; i < MR; ++i) {
    for (int j = 0; j < MT; ++j) {
      // A = Q_tile row i, B = K_tile col j; accumulate over k = 0..MK-1.
      const bfloat16 *pA = Q + i * MK * R_MAC * S_MAC;
      const bfloat16 *pB = K + j * S_MAC * T_MAC;  // K tiled as [MK][MT][s][t]
      aie::vector<bfloat16, R_MAC * S_MAC> A = aie::load_v<R_MAC * S_MAC>(pA);
      aie::vector<bfloat16, S_MAC * T_MAC> B = aie::load_v<S_MAC * T_MAC>(pB);
      MMUL C;
      C.mul(A, B);
      pA += R_MAC * S_MAC;
      pB += MT * S_MAC * T_MAC;
      for (int k = 1; k < MK; ++k) {
        A = aie::load_v<R_MAC * S_MAC>(pA); pA += R_MAC * S_MAC;
        B = aie::load_v<S_MAC * T_MAC>(pB); pB += MT * S_MAC * T_MAC;
        C.mac(A, B);
      }
      aie::store_v(&S_tiled[(i * MT + j) * R_MAC * T_MAC], C.template to_vector<float>());
    }
  }

  // Scale by inv_sqrt_d and convert to bf16 row-major S[BR][BC]. Each 4×8 tile
  // is 32 consecutive fp32 values stored as [row0..3][col0..7] row-major
  // within the tile. We linearise back into row-major S.
  alignas(64) bfloat16 S[BR * BC];
  const aie::vector<float, R_MAC * T_MAC> scale_v =
      aie::broadcast<float, R_MAC * T_MAC>(INV_SQRT_D);
  for (int i = 0; i < MR; ++i) {
    for (int j = 0; j < MT; ++j) {
      aie::vector<float, R_MAC * T_MAC> tile =
          aie::load_v<R_MAC * T_MAC>(&S_tiled[(i * MT + j) * R_MAC * T_MAC]);
      aie::accum<accfloat, R_MAC * T_MAC> scaled = aie::mul(tile, scale_v);
      aie::vector<bfloat16, R_MAC * T_MAC> bf = scaled.to_vector<bfloat16>();
      // Scatter 4×8 tile into row-major S[BR, BC]
      alignas(64) bfloat16 tmp[R_MAC * T_MAC];
      aie::store_v(&tmp[0], bf);
      for (int rr = 0; rr < R_MAC; ++rr) {
        for (int tt = 0; tt < T_MAC; ++tt) {
          S[(i * R_MAC + rr) * BC + (j * T_MAC + tt)] = tmp[rr * T_MAC + tt];
        }
      }
    }
  }

  // Causal mask: set S[r, c] = -inf where start_col+c > g_start_row+r.
  if (g_causal) {
    for (int r = 0; r < BR; ++r) {
      int row_pos = g_start_row + r;
      for (int c = 0; c < BC; ++c) {
        if (start_col + c > row_pos) {
          S[r * BC + c] = (bfloat16)(-INFINITY);
        }
      }
    }
  }

  // Row-wise online softmax, vectorised over BC. Per row: 1 reduce_max,
  // 1 vector mul (scale by log2e), 1 vector sub, 1 vector exp2, 1 vector
  // store, 1 reduce_add. Alpha uses the scalar exp_scalar helper.
  const aie::vector<bfloat16, BC> log2e_v = aie::broadcast<bfloat16, BC>((bfloat16)LOG2E);
  for (int r = 0; r < BR; ++r) {
    aie::vector<bfloat16, BC> s_v = aie::load_v<BC>(&S[r * BC]);
    bfloat16 m_local_bf = aie::reduce_max(s_v);
    float m_local = (float)m_local_bf;
    float m_prev  = g_m[r];
    float m_new   = (m_prev > m_local) ? m_prev : m_local;
    if (m_new == -INFINITY) continue;

    float alpha = (m_prev == -INFINITY) ? 0.0f : exp_scalar(m_prev - m_new);

    aie::accum<accfloat, BC> scaled_acc = aie::mul(s_v, log2e_v);
    aie::vector<bfloat16, BC> m_new_scaled_v =
        aie::broadcast<bfloat16, BC>((bfloat16)(m_new * LOG2E));
    aie::accum<accfloat, BC> diff_acc = aie::sub(scaled_acc, m_new_scaled_v);
    aie::vector<bfloat16, BC> P_v = aie::exp2<bfloat16>(diff_acc.to_vector<float>());

    alignas(64) bfloat16 P_row[BC];
    aie::store_v(&P_row[0], P_v);
    aie::accum<accfloat, BC> P_acc = aie::mul(P_v, aie::broadcast<bfloat16, BC>((bfloat16)1.0f));
    float row_sum = aie::reduce_add(P_acc.to_vector<float>());
    g_l[r] = alpha * g_l[r] + row_sum;

    // O[r, :] = alpha · O[r, :] + P_row @ V. DH=64 split into two 32-lane
    // halves. alpha rescales O_prev; then BC scalar-broadcast MACs accumulate
    // P_row[c] · V[c, :] row-by-row in fp32, storing bf16 back to g_O.
    //
    // We tried replacing this with aie::mmul<4,8,8> for P·V, but the scatter
    // of P from softmax into 4×8 tiles + gather of the fp32 output back to
    // row-major ate the mmul savings (1ms, well inside noise) AND storing
    // O_new as bf16 broke top-1 precision across 4 kv-blocks. Inline
    // bf16-accum path below keeps the fp32 running sum for precision.
    const bfloat16 alpha_bf = (bfloat16)alpha;
    aie::accum<accfloat, 32> o0 = aie::mul(aie::load_v<32>(&g_O[r * DH]),       alpha_bf);
    aie::accum<accfloat, 32> o1 = aie::mul(aie::load_v<32>(&g_O[r * DH + 32]), alpha_bf);
    for (int c = 0; c < BC; ++c) {
      aie::vector<bfloat16, 32> v0 = aie::load_v<32>(&V[c * DH]);
      aie::vector<bfloat16, 32> v1 = aie::load_v<32>(&V[c * DH + 32]);
      o0 = aie::mac(o0, v0, P_row[c]);
      o1 = aie::mac(o1, v1, P_row[c]);
    }
    aie::store_v(&g_O[r * DH],      o0.to_vector<bfloat16>());
    aie::store_v(&g_O[r * DH + 32], o1.to_vector<bfloat16>());

    g_m[r] = m_new;
  }

  g_block_idx++;
}

void attn_finalise(bfloat16 *restrict O_out) {
  for (int r = 0; r < BR; ++r) {
    float inv_l = (g_l[r] > 0.0f) ? (1.0f / g_l[r]) : 0.0f;
    for (int d = 0; d < DH; ++d) {
      O_out[r * DH + d] = (bfloat16)((float)g_O[r * DH + d] * inv_l);
    }
  }
  g_block_idx = 0;  // ready for next dispatch
}

} // extern "C"
