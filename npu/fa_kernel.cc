// FlashAttention-2 forward (non-causal for now), streaming K/V blocks.
//
// Each dispatch processes one Q block attending to all Tk keys. The host
// streams n_kv = Tk/BC key/value block pairs through the compute tile. Running
// state (row-max, row-sum, output accumulator) lives in static tile memory
// across the per-block calls so we never materialise the [BR,Tk] intermediate.
//
// Shape: BR query rows × BC kv-block cols × DH head dim (all compile-time).
// Memory per block: Q=BR·DH·2 B, KV=2·BC·DH·2 B, state=(2·BR + BR·DH)·4 B.
// With BR=BC=32, DH=64: Q=4KB, KV=4KB, state=8.5KB → streams unbounded Tk.

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

// AIE2p math.h ships no scalar exp_scalar; aie::exp2 is the only exp primitive.
// Broadcast into a 32-lane vector, exp2, extract lane 0. Wasteful but the
// inner block loop calls exp_scalar BC+1 times per row, dwarfed by the two
// scalar matmuls.
static inline float exp_scalar(float x) {
  aie::vector<float, 32> v = aie::broadcast<float, 32>(x * LOG2E);
  aie::vector<bfloat16, 32> e = aie::exp2<bfloat16>(v);
  return (float)e.get(0);
}

// Persistent per-dispatch running state. Static so it outlives individual
// function calls; IRON orchestration calls attn_init once, attn_block n_kv
// times, attn_finalise once, all within one dispatch.
static float g_m[BR];            // row-max seen so far
static float g_l[BR];            // row-sum of exp(S - m)
alignas(64) static float g_O[BR * DH];  // output accumulator (pre-normalise)

extern "C" {

void attn_init() {
  for (int r = 0; r < BR; ++r) {
    g_m[r] = -INFINITY;
    g_l[r] = 0.0f;
  }
  for (int i = 0; i < BR * DH; ++i) g_O[i] = 0.0f;
}

// One block update. KV points to [K_block | V_block] concatenated — both
// [BC, DH] row-major — so one DMA tile delivers both halves.
void attn_block(bfloat16 *restrict Q, bfloat16 *restrict KV) {
  bfloat16 *K = KV;
  bfloat16 *V = KV + BC * DH;

  // S = Q · K^T · inv_sqrt_d, kept fp32 in scratch.
  alignas(64) float S[BR * BC];
  for (int r = 0; r < BR; ++r) {
    for (int c = 0; c < BC; ++c) {
      float acc = 0.0f;
      for (int k = 0; k < DH; ++k) {
        acc += (float)Q[r * DH + k] * (float)K[c * DH + k];
      }
      S[r * BC + c] = acc * INV_SQRT_D;
    }
  }

  // Online softmax + output update, row by row.
  for (int r = 0; r < BR; ++r) {
    // Block-local max over this KV block for row r.
    float m_local = S[r * BC];
    for (int c = 1; c < BC; ++c) {
      float v = S[r * BC + c];
      if (v > m_local) m_local = v;
    }

    float m_prev = g_m[r];
    float m_new  = (m_prev > m_local) ? m_prev : m_local;
    // If m_prev is -INFINITY (first block, unseen row), alpha = 0: O_prev is 0
    // and l_prev is 0, so the rescale is a no-op regardless. exp_scalar handles
    // (-inf - m_new) → 0 correctly but guard explicitly to stay deterministic.
    float alpha = (m_prev == -INFINITY) ? 0.0f : exp_scalar(m_prev - m_new);

    // P[r, :] = exp(S[r, :] - m_new), compute row-sum in one pass.
    float P_row[BC];
    float row_sum = 0.0f;
    for (int c = 0; c < BC; ++c) {
      float e = exp_scalar(S[r * BC + c] - m_new);
      P_row[c] = e;
      row_sum += e;
    }

    g_l[r] = alpha * g_l[r] + row_sum;

    // O[r, :] = alpha · O[r, :] + P_row · V
    for (int d = 0; d < DH; ++d) {
      float o = alpha * g_O[r * DH + d];
      for (int c = 0; c < BC; ++c) {
        o += P_row[c] * (float)V[c * DH + d];
      }
      g_O[r * DH + d] = o;
    }

    g_m[r] = m_new;
  }
}

void attn_finalise(bfloat16 *restrict O_out) {
  for (int r = 0; r < BR; ++r) {
    // All-masked row (can't happen without causal mask, but future-proof):
    // l[r] == 0 → write zeros rather than div-by-zero.
    float inv_l = (g_l[r] > 0.0f) ? (1.0f / g_l[r]) : 0.0f;
    for (int d = 0; d < DH; ++d) {
      O_out[r * DH + d] = (bfloat16)(g_O[r * DH + d] * inv_l);
    }
  }
}

} // extern "C"
