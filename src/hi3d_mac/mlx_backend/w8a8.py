"""Experimental native int8 GEMMs with signed block-Hadamard outlier handling.

Explicit opt-in only. Sparse I/O, image conditioning, softmax, norms, and the
sampler remain in their original precision. The fused quantizer avoids fp32
activation-sized temporaries. The MPP GEMM accumulates exact int32 products,
then rescales in fp32 and rounds once to fp16 after adding the bias.
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np


@lru_cache(None)
def signs(width: int):
    return mx.array(np.random.default_rng(314159 + width).choice([-1.0, 1.0], width).astype(np.float32))






_QUANT = r"""
  const uint row = threadgroup_position_in_grid.y;
  const uint lane = thread_index_in_simdgroup, warp = simdgroup_index_in_threadgroup;
  threadgroup float partial[4];
  constexpr int ITERS = (K + 255) / 256;
  float lo[ITERS], hi[ITERS];
  float maximum = 0.0f;
  #pragma clang loop unroll(full)
  for (int i = 0; i < ITERS; ++i) {
    uint c = (i * 4 + warp) * 64 + lane;
    float u = c < K ? float(x[(ulong)row*K+c]) * sign[c] : 0.0f;
    float v = c+32 < K ? float(x[(ulong)row*K+c+32]) * sign[c+32] : 0.0f;
    #pragma clang loop unroll(full)
    for (uint d = 1; d < 32; d *= 2) {
      float a = simd_shuffle_xor(u, d), b = simd_shuffle_xor(v, d);
      u = (lane & d) ? a-u : u+a;
      v = (lane & d) ? b-v : v+b;
    }
    lo[i] = (u+v) * 0.125f;
    hi[i] = (u-v) * 0.125f;
    maximum = max(maximum, max(abs(lo[i]), abs(hi[i])));
  }
  maximum = simd_max(maximum);
  if (lane == 0) partial[warp] = maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float s = max(max(max(partial[0], partial[1]), max(partial[2], partial[3])), 1e-8f) / 127.0f;
  if (warp == 0 && lane == 0) scale[row] = s;
  #pragma clang loop unroll(full)
  for (int i = 0; i < ITERS; ++i) {
    uint c = (i * 4 + warp) * 64 + lane;
    if (c < K) q[(ulong)row*K+c] = char(clamp(rint(lo[i]/s), -127.0f, 127.0f));
    if (c+32 < K) q[(ulong)row*K+c+32] = char(clamp(rint(hi[i]/s), -127.0f, 127.0f));
  }
"""

_HEADER = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"
_GEMM = r"""
  const int M = dims[0], N = dims[1], K = dims[2];
  const uint2 tg = threadgroup_position_in_grid.xy;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t*)a, dextents<int32_t, 2>(K, M));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB((device int8_t*)w, dextents<int32_t, 2>(K, N));
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.slice(0, (int)(tg.y * BM));
  auto mB = tB.slice(0, (int)(tg.x * BN));
  auto acc = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
  op.run(mA, mB, acc);
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) {
      auto idx = acc.get_multidimensional_index(i);
      const int col = tg.x * BN + idx[0], row = tg.y * BM + idx[1];
      if (row < M && col < N) out[(long)row*N+col] = half(float(acc[i]) * sa[row] * sw[col] + float(bias[col]));
    }
  }
"""


@lru_cache(None)
def _quant_kernel():
    return mx.fast.metal_kernel(name="hi3d_signed_h64_quant", input_names=["x", "sign"],
                                output_names=["q", "scale"], source=_QUANT)


@lru_cache(None)
def _gemm_kernel():
    return mx.fast.metal_kernel(name="hi3d_signed_w8a8_gemm", input_names=["a", "w", "sa", "sw", "bias", "dims"],
                                output_names=["out"], source=_GEMM, header=_HEADER)


def quantize_activation(x, multiplier=None):
    k = x.shape[-1]
    if k % 64:
        raise ValueError("signed H64 requires width divisible by 64")
    x = x.reshape(-1, k)
    m = x.shape[0]
    return _quant_kernel()(inputs=[x, signs(k) if multiplier is None else multiplier], template=[("K", k)], grid=(128, m, 1),
                           threadgroup=(128, 1, 1), output_shapes=[(m, k), (m,)], output_dtypes=[mx.int8, mx.float32])


def linear(x, w, sw, bias, tile=(64, 128), multiplier=None):
    if w.dtype != mx.int8 or sw.dtype != mx.float32:
        raise ValueError("W8A8 requires int8 weights and float32 scales")
    if w.ndim != 2 or x.shape[-1] != w.shape[1] or sw.shape != (w.shape[0],) or bias.shape != (w.shape[0],):
        raise ValueError("incompatible W8A8 projection shapes")
    bm, bn = tile
    shape = x.shape[:-1]
    k, n = x.shape[-1], w.shape[0]
    a, sa = quantize_activation(x, multiplier)
    m = a.shape[0]
    y = _gemm_kernel()(inputs=[a, w, sa, sw, bias, mx.array([m, n, k], mx.int32)],
                       template=[("BM", bm), ("BN", bn)],
                       grid=((n+bn-1)//bn*128, (m+bm-1)//bm, 1), threadgroup=(128, 1, 1),
                       output_shapes=[(m, n)], output_dtypes=[mx.float16])[0]
    return y.reshape(*shape, n)
