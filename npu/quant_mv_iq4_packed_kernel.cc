// Packed IQ4_XS dequant+matvec tile kernel.
//
// Host preprocessing keeps each 256-lane block compact but removes nonlinear
// nibble lookup from the tile:
//   [ 0: 32]  dl[8] float32, one scale per 32-lane group
//   [32:288]  q[256] int8, IQ4_NL kvalues expanded from packed nibbles
//
// The tile computes bf16 MACs after int8->float scaling.

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
#define B_BYTES 288

extern "C" {

void zero_f32_qmv_iq4_packed(float *restrict c_out) {
  zero_scalar<float, DIM_M, 1>(c_out);
}

static inline float dequant_row_block_dot_iq4_packed(
    const uint8_t *restrict blk,
    const aie::vector<bfloat16, 32> *restrict b_vecs) {
  const float *restrict dl = (const float *)blk;
  const int8_t *restrict qs = (const int8_t *)(blk + 32);

  aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
  for (int ib = 0; ib < 8; ++ib) {
    const aie::vector<int8_t, 32> q_i8 = aie::load_v<32>(qs + ib * 32);
    const aie::vector<float, 32> q_f = aie::to_float<float>(q_i8);
    const aie::vector<float, 32> scale = aie::broadcast<float, 32>(dl[ib]);
    const aie::accum<accfloat, 32> q_acc = aie::mul(q_f, scale);
    acc = aie::mac(acc, q_acc.to_vector<bfloat16>(), b_vecs[ib]);
  }
  return aie::reduce_add(acc.to_vector<float>());
}

void quant_mv_iq4_xs_packed_bf16_f32(uint8_t *restrict a_in,
                                     bfloat16 *restrict b_in,
                                     float *restrict c_out) {
  event0();
  aie::vector<bfloat16, 32> b_vecs[8];
  for (int sb = 0; sb < 8; ++sb) {
    b_vecs[sb] = aie::load_v<32>(b_in + sb * 32);
  }
  for (int row = 0; row < DIM_M; ++row) {
    const uint8_t *blk = a_in + row * B_BYTES;
    c_out[row] += dequant_row_block_dot_iq4_packed(blk, b_vecs);
  }
  event1();
}

} // extern "C"
