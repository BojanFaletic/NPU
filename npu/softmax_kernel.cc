// Thin wrapper around the stock softmax_simple_bf16 that bakes the row length
// in at compile time via -DSM_LEN=N. Exposes `softmax_row_bf16` as the only
// externally visible symbol, to be imported by the IRON program.
//
// Rationale: softmax.cc's `softmax_bf16` takes a runtime int32 size. Giving
// Peano a compile-time constant lets `elem_iters = SM_LEN / SM_VEC_LEN` be
// constant-folded and the inner loops fully unrolled if the optimizer decides.
// We also get one .o per chosen row length, keyed like the matmul .o files.

#include "../vendor/mlir-aie-src/aie_kernels/aie2p/softmax.cc"

#ifndef SM_LEN
#error "SM_LEN must be defined (row length in bf16 elements, multiple of 32)"
#endif

extern "C" {

void softmax_row_bf16(bfloat16 *restrict in, bfloat16 *restrict out) {
  softmax_simple_bf16(in, out, SM_LEN);
}

} // extern "C"
