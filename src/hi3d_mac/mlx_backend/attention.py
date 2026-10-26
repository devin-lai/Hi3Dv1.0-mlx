"""Flash attention for the fp16 transformer torso on the M5 GPU neural accelerators.

``mx.fast.scaled_dot_product_attention`` reaches 8-11 TFLOPS on Hi3D's shapes (head
dim 96, 3k-26k tokens); this kernel reaches 13.8-14.3 of the ~16 TFLOPS the M5's
``matmul2d`` units deliver for fp16. Two measured causes (``scripts/research/nax_probe.py``):

* **Launch order.** With threadgroups ordered query-block-major, every concurrently
  resident threadgroup streams the same head's K/V rows at the same moment, and
  throughput falls with sequence length (this kernel launched that way: 12.0 TFLOPS
  at 3k tokens, 5.5 at 14k; MLX's kernel 10.8 and 8.9). Here consecutive
  threadgroups take *different heads* of the same query block (grid ``x = head``),
  which keeps the rate flat: 14.3 TFLOPS at 14k tokens.
* **P in fp16.** MLX feeds the fp32 probabilities to the ``P @ V`` product; a
  float x half ``matmul2d`` runs at a quarter of the half x half rate on M5 and,
  measured against float64, is also less accurate than rounding P to fp16 first
  (2.4e-4 vs 2.2e-4 RMS relative on random inputs). CUDA FlashAttention-2, which
  the official pipeline uses, rounds P to fp16 the same way.

Arithmetic: ``S = Q K^T`` accumulates in fp32, the online softmax (running max,
``exp2``, row sums of the unrounded probabilities) runs in fp32, ``P`` is rounded to
fp16 for ``P @ V``, which accumulates in fp32; the output is ``O / l`` rounded once
to fp16.

Each simdgroup owns 16 query rows (kept in registers), a threadgroup 4 simdgroups;
keys and values stream in blocks of 32 rows. Output is written as ``[B, Nq, H * D]``,
the layout the attention output projection consumes, so no transpose follows.
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

typedef metal::vec<half, 8> hfrag;   // this lane's 8 elements of a 16x16 fragment
typedef metal::vec<float, 8> ffrag;

// c[16x32] += a[16x16] * b[16x32] on the neural accelerator. b is given as two 16x16
// fragments (columns 0-15 and 16-31); with TB they are fragments of b^T (32 x 16).
template <bool TB>
inline void mma_16x32x16(thread ffrag& c0, thread ffrag& c1, thread const hfrag& a,
                         thread const hfrag& b0, thread const hfrag& b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, TB, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto cc = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; ++i) {
    ca[i] = a[i];
    cb[i] = b0[i];
    cb[8 + i] = b1[i];
    cc[i] = c0[i];
    cc[8 + i] = c1[i];
  }
  op.run(ca, cb, cc);
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; ++i) {
    c0[i] = cc[i];
    c1[i] = cc[8 + i];
  }
}

// Fragment layout of the 16x16 operands: lane l holds rows fm, fm + 8 and columns
// fn .. fn + 3, with q = l >> 2, fm = (q & 4) | ((l >> 1) & 3), fn = ((q & 2) | (l & 1)) * 4.
inline hfrag load_frag(const device half* src, int ld, short fm, short fn) {
  const half4 a = *(const device half4*)(src + fm * ld + fn);
  const half4 c = *(const device half4*)(src + (fm + 8) * ld + fn);
  return hfrag(a, c);
}

// Same, rows >= lim read as zero (last, partial block).
inline hfrag load_frag_rows(const device half* src, int ld, short fm, short fn, int lim) {
  const half4 z = half4(0.0h);
  const half4 a = fm < lim ? *(const device half4*)(src + fm * ld + fn) : z;
  const half4 c = fm + 8 < lim ? *(const device half4*)(src + (fm + 8) * ld + fn) : z;
  return hfrag(a, c);
}
"""

_SOURCE = r"""
  // grid of threadgroups: (H, ceil(Nq / 64), B); 4 simdgroups x 16 query rows each.
  constexpr int BQ = 64;
  constexpr int BK = 32;
  constexpr int TD = D / 16;
  constexpr int TK = BK / 16;
  const int Nq = params[0];
  const int Nk = params[1];
  const long kv_batch = params[2];   // 0: K/V shared by all batch elements
  const long q_row = params[3];      // element strides of q: row, head, batch
  const long q_head = params[4];
  const long q_batch = params[5];
  const float scale2 = scale[0] * 1.44269504089f;  // softmax in the exp2 domain

  const uint lane = thread_index_in_simdgroup;
  const uint3 tg = threadgroup_position_in_grid;
  const int h = tg.x;
  const int b = tg.z;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int q0 = tg.y * BQ + simdgroup_index_in_threadgroup * 16;
  if (q0 >= Nq) return;

  const device half* Q = q + b * q_batch + h * q_head + q0 * q_row;
  const device half* K = k + b * kv_batch + (long)h * Nk * D;
  const device half* V = v + b * kv_batch + (long)h * Nk * D;

  hfrag qf[TD];
  #pragma clang loop unroll(full)
  for (short d = 0; d < TD; ++d) qf[d] = load_frag_rows(Q + d * 16, q_row, fm, fn, Nq - q0);
  ffrag of[TD];
  #pragma clang loop unroll(full)
  for (short d = 0; d < TD; ++d) of[d] = ffrag(0.0f);
  float m_run[2] = {-INFINITY, -INFINITY};
  float l_run[2] = {0.0f, 0.0f};

  const int nfull = Nk / BK;
  for (int kb = 0; kb <= nfull; ++kb) {
    const int k0 = kb * BK;
    const int klim = Nk - k0;  // valid keys in this block (BK or fewer)
    const bool last = kb == nfull;
    if (last && klim <= 0) break;

    // S = Q K^T (fp32)
    ffrag sf[TK];
    #pragma clang loop unroll(full)
    for (short j = 0; j < TK; ++j) sf[j] = ffrag(0.0f);
    if (!last) {
      #pragma clang loop unroll(full)
      for (short j = 0; j < TK; j += 2) {
        #pragma clang loop unroll(full)
        for (short d = 0; d < TD; ++d) {
          const device half* kp = K + (long)(k0 + j * 16) * D + d * 16;
          mma_16x32x16<true>(sf[j], sf[j + 1], qf[d], load_frag(kp, D, fm, fn), load_frag(kp + 16 * D, D, fm, fn));
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (short j = 0; j < TK; j += 2) {
        #pragma clang loop unroll(full)
        for (short d = 0; d < TD; ++d) {
          const device half* kp = K + (long)(k0 + j * 16) * D + d * 16;
          mma_16x32x16<true>(sf[j], sf[j + 1], qf[d], load_frag_rows(kp, D, fm, fn, klim - j * 16),
                             load_frag_rows(kp + 16 * D, D, fm, fn, klim - j * 16 - 16));
        }
      }
    }

    // online softmax: this lane's rows fm and fm + 8; row partners differ in lane bits 0, 3
    float m_new[2] = {m_run[0], m_run[1]};
    #pragma clang loop unroll(full)
    for (short j = 0; j < TK; ++j) {
      #pragma clang loop unroll(full)
      for (short e = 0; e < 8; ++e) {
        float s = sf[j][e] * scale2;
        if (last && j * 16 + fn + (e & 3) >= klim) s = -INFINITY;
        sf[j][e] = s;
        m_new[e >> 2] = metal::max(m_new[e >> 2], s);
      }
    }
    float factor[2];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 2; ++i) {
      m_new[i] = metal::max(m_new[i], simd_shuffle_xor(m_new[i], ushort(1)));
      m_new[i] = metal::max(m_new[i], simd_shuffle_xor(m_new[i], ushort(8)));
      factor[i] = fast::exp2(m_run[i] - m_new[i]);
      m_run[i] = m_new[i];
    }
    float rsum[2] = {0.0f, 0.0f};
    hfrag pf[TK];
    #pragma clang loop unroll(full)
    for (short j = 0; j < TK; ++j) {
      #pragma clang loop unroll(full)
      for (short e = 0; e < 8; ++e) {
        const float p = fast::exp2(sf[j][e] - m_run[e >> 2]);
        rsum[e >> 2] += p;
        pf[j][e] = half(p);
      }
    }
    #pragma clang loop unroll(full)
    for (short i = 0; i < 2; ++i) {
      rsum[i] += simd_shuffle_xor(rsum[i], ushort(1));
      rsum[i] += simd_shuffle_xor(rsum[i], ushort(8));
      l_run[i] = l_run[i] * factor[i] + rsum[i];
    }
    #pragma clang loop unroll(full)
    for (short d = 0; d < TD; ++d) {
      #pragma clang loop unroll(full)
      for (short e = 0; e < 8; ++e) of[d][e] *= factor[e >> 2];
    }

    // O += P V (fp16 P, fp32 accumulation)
    if (!last) {
      #pragma clang loop unroll(full)
      for (short d = 0; d < TD; d += 2) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < TK; ++j) {
          const device half* vp = V + (long)(k0 + j * 16) * D + d * 16;
          mma_16x32x16<false>(of[d], of[d + 1], pf[j], load_frag(vp, D, fm, fn), load_frag(vp + 16, D, fm, fn));
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (short d = 0; d < TD; d += 2) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < TK; ++j) {
          const device half* vp = V + (long)(k0 + j * 16) * D + d * 16;
          mma_16x32x16<false>(of[d], of[d + 1], pf[j], load_frag_rows(vp, D, fm, fn, klim - j * 16),
                              load_frag_rows(vp + 16, D, fm, fn, klim - j * 16));
        }
      }
    }
  }

  // O / l as [B, Nq, H * D]
  #pragma clang loop unroll(full)
  for (short i = 0; i < 2; ++i) {
    const int row = q0 + fm + 8 * i;
    if (row < Nq) {
      const float inv = 1.0f / l_run[i];
      device half* dst = out + ((long)b * Nq + row) * (H * D) + h * D + fn;
      #pragma clang loop unroll(full)
      for (short d = 0; d < TD; ++d) {
        *(device half4*)(dst + d * 16) = half4(
            of[d][4 * i] * inv, of[d][4 * i + 1] * inv, of[d][4 * i + 2] * inv, of[d][4 * i + 3] * inv);
      }
    }
  }
"""

_KERNEL = None


def flash_attention(q: mx.array, k: mx.array, v: mx.array, scale: float, q_layout: str = "bhnd") -> mx.array:
    """``softmax(q k^T * scale) v`` for fp16 inputs, returned as ``[B, Nq, H * D]`` fp16.

    Args:
        q: ``[B, H, Nq, D]`` (``q_layout="bhnd"``) or ``[B, Nq, H, D]`` (``"bnhd"``).
        k, v: ``[B, H, Nk, D]``, or ``[1, H, Nk, D]`` shared by every batch element.
        scale: Softmax scale applied to ``q k^T``.
    """
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="hi3d_flash_attention_nax",
            input_names=["q", "k", "v", "params", "scale"],
            output_names=["out"],
            source=_SOURCE,
            header=_HEADER,
        )
    if q_layout == "bhnd":
        batch, heads, nq, dim = q.shape
        strides = (dim, nq * dim, heads * nq * dim)
    elif q_layout == "bnhd":
        batch, nq, heads, dim = q.shape
        strides = (heads * dim, dim, nq * heads * dim)
    else:
        raise ValueError(f"unknown q layout {q_layout!r}")
    nk = k.shape[2]
    if dim % 32 or k.shape[1] != heads or k.shape[3] != dim or k.shape[0] not in (1, batch):
        raise ValueError(f"unsupported shapes q {q.shape} k {k.shape}")
    if q.dtype != mx.float16 or k.dtype != mx.float16 or v.dtype != mx.float16:
        raise ValueError("flash_attention takes float16 inputs")
    kv_batch = 0 if k.shape[0] == 1 else heads * nk * dim
    params = mx.array([nq, nk, kv_batch, *strides], dtype=mx.int32)
    return _KERNEL(
        inputs=[q, k, v, params, mx.array([scale], dtype=mx.float32)],
        template=[("H", heads), ("D", dim)],
        grid=(heads * 128, -(-nq // 64), batch),
        threadgroup=(128, 1, 1),
        output_shapes=[(batch, nq, heads * dim)],
        output_dtypes=[mx.float16],
    )[0]
