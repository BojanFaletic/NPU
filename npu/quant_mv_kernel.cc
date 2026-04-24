// IQ3_XXS dequant+matvec tile kernel.
//
// Computes  c[m] += sum over the K_BLOCKS K-blocks of  W_dequant[m, k] * b[k]
// where W is stored as IQ3_XXS bytes laid out per row, and b is bf16.
//
// Per-block byte layout (host-preprocessed: original fp16 d → fp32):
//   [ 0: 4]   d (fp32)
//   [ 4:68]   qs[64]   (uint8 grid indices into 256-entry LUT)
//   [68:100]  scales_signs[8] (uint32) — 4-bit scale + 4×7-bit sign indices
// Block covers BLK = 256 K-elements; B_BYTES = 100 bytes per block.
//
// Constants (linked into the .o, placed in tile DM):
//   GRID   : (256, 4) fp32 — abs values from grid_map after 8-level decode
//   KSIGNS : (128, 8) fp32 — sign multipliers (+1 / -1) per sub-byte bit
//
// DIM_M is the row-tile height (output rows processed per kernel invocation).

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>
#include <type_traits>

#include "../vendor/mlir-aie-src/aie_kernels/aie_kernel_utils.h"
#include "../vendor/mlir-aie-src/aie_kernels/aie2p/zero.cc"

#ifndef DIM_M
#define DIM_M 32
#endif

#define BLK 256
#define B_BYTES 100

// Generated tables (4 KB grid + 1 KB ksigns; included from a header that
// quant_mv.py emits into the build dir).
#include "iq3_xxs_tables.h"

extern "C" {

void zero_f32_qmv(float *restrict c_out) {
  zero_scalar<float, DIM_M, 1>(c_out);
}

// Dequantizes one block of one row, accumulating one row's K-block contribution
// into the returned partial sum.
static inline float dequant_row_block_dot(const uint8_t *restrict blk,
                                          const bfloat16 *restrict b_in) {
  float d;
  memcpy(&d, blk, 4);
  const uint8_t *qs = blk + 4;
  uint32_t scales[8];
  memcpy(scales, blk + 68, 32);

  float sum = 0.0f;
  for (int sb = 0; sb < 8; ++sb) {
    const uint32_t sw = scales[sb];
    const float db = d * (0.5f + (float)((sw >> 28) & 0xF)) * 0.5f;
    for (int g = 0; g < 4; ++g) {
      const uint32_t sign_idx = (sw >> (g * 7)) & 0x7F;
      const float *signs = IQ3_XXS_KSIGNS_FP + sign_idx * 8;
      const uint8_t qa = qs[sb * 8 + g * 2];
      const uint8_t qb = qs[sb * 8 + g * 2 + 1];
      const float *ga = IQ3_XXS_GRID + qa * 4;
      const float *gb = IQ3_XXS_GRID + qb * 4;
      const int lane0 = sb * 32 + g * 8;
      sum += db * signs[0] * ga[0] * (float)b_in[lane0 + 0];
      sum += db * signs[1] * ga[1] * (float)b_in[lane0 + 1];
      sum += db * signs[2] * ga[2] * (float)b_in[lane0 + 2];
      sum += db * signs[3] * ga[3] * (float)b_in[lane0 + 3];
      sum += db * signs[4] * gb[0] * (float)b_in[lane0 + 4];
      sum += db * signs[5] * gb[1] * (float)b_in[lane0 + 5];
      sum += db * signs[6] * gb[2] * (float)b_in[lane0 + 6];
      sum += db * signs[7] * gb[3] * (float)b_in[lane0 + 7];
    }
  }
  return sum;
}

// One-block matvec tile entry point. Inputs are sized for DIM_M rows × 1
// K-block (BLK K-elements). The MLIR runtime calls this once per K-block per
// row tile, accumulating into c_out. zero_f32_qmv must be called first to
// clear the accumulator at the start of each row tile.
void quant_mv_iq3_xxs_bf16_f32(uint8_t *restrict a_in,
                               bfloat16 *restrict b_in,
                               float *restrict c_out) {
  event0();
  for (int row = 0; row < DIM_M; ++row) {
    const uint8_t *blk = a_in + row * B_BYTES;
    c_out[row] += dequant_row_block_dot(blk, b_in);
  }
  event1();
}

} // extern "C"
