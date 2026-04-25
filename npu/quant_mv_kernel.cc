// IQ3_XXS dequant+matvec tile kernel.
//
// Computes  c[m] += sum over the K_BLOCKS K-blocks of  W_dequant[m, k] * b[k]
// where W is stored as IQ3_XXS bytes laid out per row, and b is bf16.
//
// Per-block byte layout (host-preprocessed):
//   [  0: 32] db[8]     (fp32 per 32-lane sub-block scale)
//   [ 32: 96] qs[64]    (uint8 grid indices into 256-entry LUT)
//   [ 96:128] signs[32] (uint8 7-bit sign LUT indices; 8 sub-blocks × 4 groups)
// Block covers BLK = 256 K-elements; B_BYTES = 128 bytes per block.
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
#define B_BYTES 128

// Generated tables (4 KB grid + 1 KB ksigns; included from a header that
// quant_mv.py emits into the build dir).
#include "iq3_xxs_tables.h"

// Builds one 8-lane IQ3 group with explicit vector ops, then inserts it into
// the 32-lane sub-block consumed by the native bf16 MAC.
template <int G>
static inline void set_dequant_group(aie::vector<bfloat16, 32> &qv,
                                     const uint8_t *restrict qs,
                                     const uint8_t *restrict sign_idxs,
                                     float db) {
  const float *signs = IQ3_XXS_KSIGNS_FP + sign_idxs[G] * 8;
  const uint8_t qa = qs[G * 2];
  const uint8_t qb = qs[G * 2 + 1];
  const float *ga = IQ3_XXS_GRID + qa * 4;
  const float *gb = IQ3_XXS_GRID + qb * 4;

  const aie::vector<float, 8> signs_v = aie::load_v<8>(signs);
  const aie::vector<float, 4> ga_v = aie::load_v<4>(ga);
  const aie::vector<float, 4> gb_v = aie::load_v<4>(gb);
  aie::vector<float, 8> grid_v;
  grid_v.insert(0, ga_v);
  grid_v.insert(1, gb_v);
  const aie::vector<float, 8> sg =
      aie::mul(signs_v, grid_v).to_vector<float>();
  const aie::vector<float, 8> db_v = aie::broadcast<float, 8>(db);
  const aie::accum<accfloat, 8> q_acc = aie::mul(sg, db_v);
  qv.insert(G, q_acc.to_vector<bfloat16>());
}

extern "C" {

void zero_f32_qmv(float *restrict c_out) {
  zero_scalar<float, DIM_M, 1>(c_out);
}

// Dequantizes one block of one row, accumulating one row's K-block contribution
// into the returned partial sum.
static inline float dequant_row_block_dot(
    const uint8_t *restrict blk,
    const aie::vector<bfloat16, 32> *restrict b_vecs) {
  const uint8_t *qs = blk + 32;
  const uint8_t *sign_idxs = blk + 96;
  const float *dbs = (const float *)blk;

  aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
  for (int sb = 0; sb < 8; ++sb) {
    const float db = dbs[sb];
    aie::vector<bfloat16, 32> q_bf;
    const uint8_t *sb_qs = qs + sb * 8;
    const uint8_t *sb_sign_idxs = sign_idxs + sb * 4;
    set_dequant_group<0>(q_bf, sb_qs, sb_sign_idxs, db);
    set_dequant_group<1>(q_bf, sb_qs, sb_sign_idxs, db);
    set_dequant_group<2>(q_bf, sb_qs, sb_sign_idxs, db);
    set_dequant_group<3>(q_bf, sb_qs, sb_sign_idxs, db);
    acc = aie::mac(acc, q_bf, b_vecs[sb]);
  }
  return aie::reduce_add(acc.to_vector<float>());
}

// One-block matvec tile entry point. Inputs are sized for DIM_M rows × 1
// K-block (BLK K-elements). The MLIR runtime calls this once per K-block per
// row tile, accumulating into c_out. zero_f32_qmv must be called first to
// clear the accumulator at the start of each row tile.
void quant_mv_iq3_xxs_bf16_f32(uint8_t *restrict a_in,
                               bfloat16 *restrict b_in,
                               float *restrict c_out) {
  event0();
  aie::vector<bfloat16, 32> b_vecs[8];
  for (int sb = 0; sb < 8; ++sb) {
    b_vecs[sb] = aie::load_v<32>(b_in + sb * 32);
  }
  for (int row = 0; row < DIM_M; ++row) {
    const uint8_t *blk = a_in + row * B_BYTES;
    c_out[row] += dequant_row_block_dot(blk, b_vecs);
  }
  event1();
}

} // extern "C"
