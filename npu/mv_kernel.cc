// bf16 matrix-vector tile kernel for decode linears.
//
// Computes C[M] += A[M,K] @ b[K] for one output-row tile. A is row-major.

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

#include "../vendor/mlir-aie-src/aie_kernels/aie_kernel_utils.h"
#include "../vendor/mlir-aie-src/aie_kernels/aie2p/zero.cc"

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

extern "C" {

void zero_f32(float *restrict c_out) {
  zero_scalar<float, DIM_M, 1>(c_out);
}

void matvec_bf16_f32(bfloat16 *restrict a_in, bfloat16 *restrict b_in,
                     float *restrict c_out) {
  event0();
  constexpr int VEC = 32;
  static_assert(DIM_K % VEC == 0);

  for (int row = 0; row < DIM_M; ++row) {
    float sum = 0.0f;
    bfloat16 *restrict a_ptr = a_in + row * DIM_K;
    bfloat16 *restrict b_ptr = b_in;
    for (int col = 0; col < DIM_K; col += VEC) {
      aie::vector<bfloat16, VEC> a_vec = aie::load_v<VEC>(a_ptr);
      aie::vector<bfloat16, VEC> b_vec = aie::load_v<VEC>(b_ptr);
      aie::accum<accfloat, VEC> prod = aie::mul(a_vec, b_vec);
      sum += aie::reduce_add(prod.to_vector<float>());
      a_ptr += VEC;
      b_ptr += VEC;
    }
    c_out[row] += sum;
  }
  event1();
}

} // extern "C"
