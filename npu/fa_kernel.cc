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
// which the end-to-end self-test accepts at ~1e-2.
static int   g_block_idx = 0;
static int   g_start_row = 0;
static int   g_causal    = 0;
static float g_m[BR];
static float g_l[BR];
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

  // S = Q · K^T · inv_sqrt_d. DH=64 splits into two 32-lane vector halves
  // (512-bit is the native AIE vector width), each r/c pair: 2 loads per operand,
  // 1 mul + 1 mac + reduce_add. 1024 vector ops total vs 65K scalar.
  alignas(64) float S[BR * BC];
  for (int r = 0; r < BR; ++r) {
    aie::vector<bfloat16, 32> q0 = aie::load_v<32>(&Q[r * DH]);
    aie::vector<bfloat16, 32> q1 = aie::load_v<32>(&Q[r * DH + 32]);
    for (int c = 0; c < BC; ++c) {
      aie::vector<bfloat16, 32> k0 = aie::load_v<32>(&K[c * DH]);
      aie::vector<bfloat16, 32> k1 = aie::load_v<32>(&K[c * DH + 32]);
      aie::accum<accfloat, 32> acc = aie::mul(q0, k0);
      acc = aie::mac(acc, q1, k1);
      S[r * BC + c] = aie::reduce_add(acc.to_vector<float>()) * INV_SQRT_D;
    }
  }

  // Causal mask: row r (abs pos g_start_row+r) can attend to col c
  // (abs pos start_col+c) iff start_col+c <= g_start_row+r. Masked elements
  // become -INFINITY so they contribute 0 to the row-max and to exp downstream.
  if (g_causal) {
    for (int r = 0; r < BR; ++r) {
      int row_pos = g_start_row + r;
      for (int c = 0; c < BC; ++c) {
        if (start_col + c > row_pos) {
          S[r * BC + c] = -INFINITY;
        }
      }
    }
  }

  for (int r = 0; r < BR; ++r) {
    float m_local = S[r * BC];
    for (int c = 1; c < BC; ++c) {
      float v = S[r * BC + c];
      if (v > m_local) m_local = v;
    }
    float m_prev = g_m[r];
    float m_new  = (m_prev > m_local) ? m_prev : m_local;
    // Fully-masked row this block AND no prior keys: leave state untouched.
    // Can happen under causal mask when start_col > row_pos (block entirely
    // past the causal frontier for this row).
    if (m_new == -INFINITY) continue;

    float alpha = (m_prev == -INFINITY) ? 0.0f : exp_scalar(m_prev - m_new);

    // P_row = exp(S[r, :] - m_new). Keep both bf16 (for the S·V vector MAC)
    // and a running fp32 sum for l[r].
    alignas(64) bfloat16 P_row[BC];
    float row_sum = 0.0f;
    for (int c = 0; c < BC; ++c) {
      float e = exp_scalar(S[r * BC + c] - m_new);
      P_row[c] = (bfloat16)e;
      row_sum += e;
    }
    g_l[r] = alpha * g_l[r] + row_sum;

    // O[r, :] = alpha · O[r, :] + P_row @ V. Output row spans DH=64 lanes,
    // split into two 32-lane halves. alpha scales the prior O accumulator
    // via a vector mul; each P_row[c] is then a bf16 scalar broadcast-MAC
    // against one row of V (both halves).
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
