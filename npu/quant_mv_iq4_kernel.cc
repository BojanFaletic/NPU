// IQ4_XS dequant+matvec tile kernel.
//
// Computes c[m] += sum over one 256-element K block of W_dequant[m, k] * b[k].
// The host preprocesses each on-disk IQ4_XS block into 256 bf16 dequantized
// values. This first IQ4 path is intentionally speed-oriented: it spends more
// resident weight memory for routed down experts so the AIE tile can use native
// 32-lane bf16 MACs instead of scalar nonlinear-nibble lookup.

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

#include "../vendor/mlir-aie-src/aie_kernels/aie_kernel_utils.h"
#include "../vendor/mlir-aie-src/aie_kernels/aie2p/zero.cc"

#ifndef DIM_M
#define DIM_M 32
#endif

#define BLK 256
#define B_BYTES 512

extern "C" {

void zero_f32_qmv_iq4(float *restrict c_out) {
  zero_scalar<float, DIM_M, 1>(c_out);
}

static inline float dequant_row_block_dot_iq4(const uint8_t *restrict blk,
                                              const bfloat16 *restrict b_in) {
  const bfloat16 *w = (const bfloat16 *)blk;

  aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
  for (int col = 0; col < BLK; col += 32) {
    const aie::vector<bfloat16, 32> w_v = aie::load_v<32>(w + col);
    const aie::vector<bfloat16, 32> b_v = aie::load_v<32>(b_in + col);
    acc = aie::mac(acc, w_v, b_v);
  }
  return aie::reduce_add(acc.to_vector<float>());
}

void quant_mv_iq4_xs_bf16_f32(uint8_t *restrict a_in,
                              bfloat16 *restrict b_in,
                              float *restrict c_out) {
  event0();
  for (int row = 0; row < DIM_M; ++row) {
    const uint8_t *blk = a_in + row * B_BYTES;
    c_out[row] += dequant_row_block_dot_iq4(blk, b_in);
  }
  event1();
}

} // extern "C"
