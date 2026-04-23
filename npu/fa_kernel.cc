// Fused attention on AIE2p: single-block softmax(Q·K^T / sqrt(D)) · V.
//
// Shapes (compile-time):
//   Q [TQ, D] bf16, K [TK, D] bf16, V [TK, D] bf16 -> O [TQ, D] bf16
// DMEM budget for TQ=TK=32, D=64: Q/K/V = 4KB each, S = 2KB, O (caller) = 4KB.
// No online rescaling — the single [TQ,TK] S tile sits in tile memory whole.

#define NOCPP
#include <aie_api/aie.hpp>
#include <stdint.h>

// Pull in softmax_simple_bf16 from the stock kernel. This file also defines
// extern "C" softmax_bf16, harmless in our .o since nothing else links it.
#include "../vendor/mlir-aie-src/aie_kernels/aie2p/softmax.cc"

#ifndef FA_TQ
#define FA_TQ 32
#endif
#ifndef FA_TK
#define FA_TK 32
#endif
#ifndef FA_D
#define FA_D 64
#endif

constexpr int TQ = FA_TQ;
constexpr int TK = FA_TK;
constexpr int DH = FA_D;

static_assert(TK % 32 == 0, "TK must be a multiple of the softmax vector width (32)");

// Input layout: one contiguous buffer [Q || K || V], offsets fixed by
// compile-time shapes. Packing is host-side, cheaper than asking for 3 input
// DMA channels (compute tiles have a 2-channel limit on AIE2p).
constexpr int QKV_LEN = TQ * DH + TK * DH + TK * DH;

extern "C" {

void attention_bf16(
    bfloat16 *restrict QKV, // [TQ*DH + TK*DH + TK*DH]  row-major, packed Q|K|V
    bfloat16 *restrict O    // [TQ, DH]  row-major (output)
) {
  event0();

  bfloat16 *Q = QKV;
  bfloat16 *K = QKV + TQ * DH;
  bfloat16 *V = QKV + TQ * DH + TK * DH;

  // S = Q·K^T / sqrt(DH) in bf16, row-major [TQ, TK]. alignas for the
  // vector iterators inside softmax_simple_bf16.
  alignas(64) bfloat16 S[TQ * TK];

  // softmax_simple_bf16 applies its own log2e scaling; don't pre-apply it.
  static_assert(DH == 64, "inv_sqrt_d hard-coded for DH=64; update for other head dims");
  const float inv_sqrt_d = 0.125f;  // 1/sqrt(64)

  // Step 1: S[i,j] = Q[i,:]·K[j,:] * inv_sqrt_d
  for (int i = 0; i < TQ; ++i) {
    for (int j = 0; j < TK; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < DH; ++k) {
        acc += (float)Q[i * DH + k] * (float)K[j * DH + k];
      }
      S[i * TK + j] = (bfloat16)(acc * inv_sqrt_d);
    }
  }

  // Step 2: per-row softmax in place. S now holds attention probabilities P.
  for (int r = 0; r < TQ; ++r) {
    softmax_simple_bf16(&S[r * TK], &S[r * TK], TK);
  }

  // Step 3: O[i,d] = sum_k P[i,k] * V[k,d]
  for (int i = 0; i < TQ; ++i) {
    for (int d = 0; d < DH; ++d) {
      float acc = 0.0f;
      for (int k = 0; k < TK; ++k) {
        acc += (float)S[i * TK + k] * (float)V[k * DH + d];
      }
      O[i * DH + d] = (bfloat16)acc;
    }
  }

  event1();
}

} // extern "C"
