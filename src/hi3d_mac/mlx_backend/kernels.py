"""Custom Metal kernels (``mx.fast.metal_kernel``) for passes MLX runs as several
memory-bound ops.

* :func:`qkv_rope_rms`: RoPE + per-head QK RMSNorm + the SDPA head layout, straight
  from the fused QKV projection of a transformer block.
* :func:`layer_norm_modulate`: LayerNorm with fp32 statistics, optionally followed
  by the adaLN modulation, one pass over fp16 rows.
* :func:`gather_group_norm_swish`: the input block of a sparse convolution that
  follows ``swish(GroupNorm(x))`` in the VAE, gathered and activated in one pass.

Each keeps the arithmetic of the composed MLX ops; only fp32 summation order of
reductions can differ.
"""

from __future__ import annotations

import mlx.core as mx


# ----------------------------------------------------------------------------- RoPE + RMSNorm

_QKV_SOURCE = """
    uint ph = thread_position_in_grid.x;   // part * H + head
    uint m = thread_position_in_grid.y;    // row of the [B * N] token batch
    if (m >= M || ph >= 3 * H) return;
    uint part = ph / H;
    uint h = ph % H;
    uint b = m / N;
    uint n = m % N;
    typedef vec<T, 2> T2;
    const device T2* src = (const device T2*)(qkv + ((ulong)(m * 3 + part) * H + h) * D);
    ulong dst_off = (((ulong)b * H + h) * N + n) * D;
    if (part == 2) {
        device T2* dst = (device T2*)(v + dst_off);
        for (uint i = 0; i < D / 2; ++i) dst[i] = src[i];
        return;
    }
    const device float* c = cos + (ulong)n * (D / 2);
    const device float* s = sin + (ulong)n * (D / 2);
    const device float* g = (part == 0 ? gq : gk) + h * D;
    float ss = 0.0f;
    for (uint i = 0; i < D / 2; ++i) {
        float2 x = float2(src[i]);
        float re = x.x * c[i] - x.y * s[i];
        float im = x.x * s[i] + x.y * c[i];
        ss += re * re + im * im;
    }
    float nrm = max(sqrt(ss), 1e-12f);
    const float scale = sqrt(float(D));
    device T2* dst = (device T2*)((part == 0 ? q : k) + dst_off);
    for (uint i = 0; i < D / 2; ++i) {
        float2 x = float2(src[i]);
        float re = x.x * c[i] - x.y * s[i];
        float im = x.x * s[i] + x.y * c[i];
        dst[i] = T2(T((re / nrm) * g[2 * i] * scale), T((im / nrm) * g[2 * i + 1] * scale));
    }
"""

_QKV_KERNEL = None


def qkv_rope_rms(qkv: mx.array, cos: mx.array, sin: mx.array, gamma_q: mx.array, gamma_k: mx.array,
                 batch: int, heads: int, head_dim: int):
    """Split a fused ``[B * N, 3 * H * D]`` projection into SDPA-ready heads.

    Applies the spatial RoPE (adjacent channel pairs, ``cos``/``sin`` ``[N, D/2]``)
    and the per-head RMS normalisation ``x / max(|x|, 1e-12) * gamma * sqrt(D)`` to
    queries and keys in float32, as ``SparseMultiHeadAttention`` does, and writes
    ``q, k, v`` as ``[B, H, N, D]`` in the projection's dtype (float16 or float32)
    in one pass (no separate rotation, normalisation and transpose kernels).
    """
    global _QKV_KERNEL
    if _QKV_KERNEL is None:
        _QKV_KERNEL = mx.fast.metal_kernel(
            name="hi3d_qkv_rope_rms",
            input_names=["qkv", "cos", "sin", "gq", "gk"],
            output_names=["q", "k", "v"],
            source=_QKV_SOURCE,
        )
    rows = qkv.shape[0]
    n = rows // batch
    shape = (batch, heads, n, head_dim)
    return _QKV_KERNEL(
        inputs=[qkv, cos, sin, gamma_q, gamma_k],
        template=[("T", qkv.dtype), ("M", rows), ("N", n), ("H", heads), ("D", head_dim)],
        grid=(3 * heads, rows, 1),
        threadgroup=(3 * heads, max(1, 512 // (3 * heads)), 1),
        output_shapes=[shape, shape, shape],
        output_dtypes=[qkv.dtype] * 3,
    )


_QKV_SIMD_SOURCE = r"""
    uint ph = thread_position_in_grid.x / GROUP;
    uint lane = thread_position_in_grid.x % GROUP;
    uint m = thread_position_in_grid.y;
    if (m >= M || ph >= 3 * H) return;
    uint part = ph / H, h = ph % H, b = m / N, n = m % N;
    typedef vec<T, 2> T2;
    const device T2* src = (const device T2*)(qkv + ((ulong)(m * 3 + part) * H + h) * D);
    ulong dst_off = (((ulong)b * H + h) * N + n) * D;
    if (part == 2) {
        device T2* dst = (device T2*)(v + dst_off);
        for (uint i = lane; i < D / 2; i += GROUP) dst[i] = src[i];
        return;
    }
    const device float* c = cos + (ulong)n * (D / 2);
    const device float* s = sin + (ulong)n * (D / 2);
    const device float* g = (part == 0 ? gq : gk) + h * D;
    float2 rotated[(D / 2 + GROUP - 1) / GROUP];
    float ss = 0.0f;
    for (uint j = 0; j < (D / 2 + GROUP - 1) / GROUP; ++j) {
        uint i = lane + j * GROUP;
        float2 r = float2(0.0f);
        if (i < D / 2) {
            float2 x = float2(src[i]);
            r = float2(x.x * c[i] - x.y * s[i], x.x * s[i] + x.y * c[i]);
            ss += r.x * r.x + r.y * r.y;
        }
        rotated[j] = r;
    }
    for (ushort offset = GROUP / 2; offset > 0; offset /= 2) ss += simd_shuffle_xor(ss, offset);
    float nrm = max(sqrt(ss), 1e-12f);
    const float scale = sqrt(float(D));
    device T2* dst = (device T2*)((part == 0 ? q : k) + dst_off);
    for (uint j = 0; j < (D / 2 + GROUP - 1) / GROUP; ++j) {
        uint i = lane + j * GROUP;
        if (i < D / 2) {
            float2 r = rotated[j];
            dst[i] = T2(T((r.x / nrm) * g[2 * i] * scale), T((r.y / nrm) * g[2 * i + 1] * scale));
        }
    }
"""
_QKV_SIMD_KERNEL = None


def qkv_rope_rms_simd(qkv, cos, sin, gamma_q, gamma_k, batch, heads, head_dim, group=16):
    """Coalesced version of :func:`qkv_rope_rms`, one SIMD subgroup per head.

    Rotates each pair once and distributes the FP32 norm reduction across lanes.
    Only FP32 reduction order changes; the value tensor remains bitwise identical.
    """
    global _QKV_SIMD_KERNEL
    if group not in (8, 16, 32) or head_dim % 2:
        raise ValueError("an even head dimension and subgroup width 8, 16 or 32 are required")
    if _QKV_SIMD_KERNEL is None:
        _QKV_SIMD_KERNEL = mx.fast.metal_kernel(
            name="hi3d_qkv_rope_rms_simd", input_names=["qkv", "cos", "sin", "gq", "gk"],
            output_names=["q", "k", "v"], source=_QKV_SIMD_SOURCE)
    rows = qkv.shape[0]
    n = rows // batch
    shape = (batch, heads, n, head_dim)
    return _QKV_SIMD_KERNEL(
        inputs=[qkv, cos, sin, gamma_q, gamma_k],
        template=[("T", qkv.dtype), ("M", rows), ("N", n), ("H", heads), ("D", head_dim), ("GROUP", group)],
        grid=(3 * heads * group, rows, 1), threadgroup=(128, 1, 1),
        output_shapes=[shape] * 3, output_dtypes=[qkv.dtype] * 3)


# ----------------------------------------------------------------------------- LayerNorm

_LN_SOURCE = """
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint warp = simdgroup_index_in_threadgroup;
    threadgroup float partial[32];
    const device half* xr = x + (ulong)row * C;
    device half* yr = out + (ulong)row * C;

    float sum = 0.0f;
    for (uint i = tid; i < C; i += TG) sum += float(xr[i]);
    sum = simd_sum(sum);
    if (lane == 0) partial[warp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint w = 0; w < TG / 32; ++w) total += partial[w];
    float mean = total / float(C);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sq = 0.0f;
    for (uint i = tid; i < C; i += TG) { float d = float(xr[i]) - mean; sq += d * d; }
    sq = simd_sum(sq);
    if (lane == 0) partial[warp] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint w = 0; w < TG / 32; ++w) var += partial[w];
    float inv = metal::precise::rsqrt(var / float(C) + eps[0]);

    for (uint i = tid; i < C; i += TG) {
        float y = (float(xr[i]) - mean) * inv;
        if (AFFINE) y = y * weight[i] + bias[i];
        half h = half(y);
        if (MODULATE) {
            // Two fp16 roundings, as the composed ops (and upstream): no FMA.
            half p = half(float(h) * float(scale[i]));
            h = half(float(p) + float(shift[i]));
        }
        yr[i] = h;
    }
"""

_LN_KERNEL = None


def layer_norm_modulate(x: mx.array, eps: float, weight: mx.array | None = None, bias: mx.array | None = None,
                        scale: mx.array | None = None, shift: mx.array | None = None) -> mx.array:
    """``LayerNorm32`` of fp16 rows (statistics in fp32), then optionally the fp16
    adaLN modulation ``y * scale + shift``, in one pass over ``x``.

    Matches ``layer_norm32(x, weight, bias) * scale + shift`` with fp32 affine
    parameters and fp16 ``scale``/``shift`` (two fp16 roundings, as the composed
    ops); only the fp32 summation order of the statistics differs.
    """
    global _LN_KERNEL
    if _LN_KERNEL is None:
        _LN_KERNEL = mx.fast.metal_kernel(
            name="hi3d_layer_norm_modulate",
            input_names=["x", "weight", "bias", "scale", "shift", "eps"],
            output_names=["out"],
            source=_LN_SOURCE,
        )
    c = x.shape[-1]
    rows = x.size // c
    dummy32 = mx.zeros((1,), mx.float32)
    dummy16 = mx.zeros((1,), mx.float16)
    threads = 256
    return _LN_KERNEL(
        inputs=[x, dummy32 if weight is None else weight, dummy32 if bias is None else bias,
                dummy16 if scale is None else scale.reshape(-1), dummy16 if shift is None else shift.reshape(-1),
                mx.array([eps], mx.float32)],
        template=[("C", c), ("TG", threads), ("AFFINE", weight is not None), ("MODULATE", scale is not None)],
        grid=(threads, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float16],
    )[0]


# ----------------------------------------------------------------------------- VAE conv prologue

_GATHER_ACT_SOURCE = """
    uint c4 = thread_position_in_grid.x;  // group of four channels
    uint j = thread_position_in_grid.y;   // row * 27 + kernel offset, within the block
    if (c4 >= C / 4 || j >= sizes[1]) return;
    int r = table[j];
    float4 v = float4(0.0f);
    if (r < sizes[0]) {
        uint c = c4 * 4;
        float4 xv = float4(*((const device vec<X, 4>*)(x + (ulong)r * C + c)));
        float4 m = float4(mean[c], mean[c + 1], mean[c + 2], mean[c + 3]);
        float4 iv = float4(inv[c], inv[c + 1], inv[c + 2], inv[c + 3]);
        float4 w = float4(weight[c], weight[c + 1], weight[c + 2], weight[c + 3]);
        float4 b = float4(bias[c], bias[c + 1], bias[c + 2], bias[c + 3]);
        float4 u = (xv - m) * iv * w + b;
        float4 e = 1.0f / (1.0f + metal::exp(metal::abs(u)));  // MLX's sigmoid
        v = u * select(1.0f - e, e, u < 0.0f);
    }
    *((device vec<T, 4>*)(out + (ulong)j * C + c4 * 4)) = vec<T, 4>(v);
"""

_GATHER_ACT_KERNEL = None


def gather_group_norm_swish(x: mx.array, table: mx.array, mean: mx.array, inv: mx.array,
                            weight: mx.array, bias: mx.array, dtype: mx.Dtype) -> mx.array:
    """Convolution input block ``swish(GN(x))[table]`` without materialising ``swish(GN(x))``.

    Computes ``u = (x[r] - mean) * inv * weight + bias`` and ``u * sigmoid(u)`` in
    float32 for every neighbour ``r = table[o, k]`` (zeros where ``r == N``: the
    convolution's zero padding applies to the activated features) and returns the
    ``[rows, 27 * C]`` block in ``dtype``.
    """
    global _GATHER_ACT_KERNEL
    if _GATHER_ACT_KERNEL is None:
        _GATHER_ACT_KERNEL = mx.fast.metal_kernel(
            name="hi3d_gather_group_norm_swish",
            input_names=["x", "table", "mean", "inv", "weight", "bias", "sizes"],
            output_names=["out"],
            source=_GATHER_ACT_SOURCE,
        )
    n, c = x.shape
    if c % 4:
        raise ValueError("channels must be a multiple of 4")
    rows = table.shape[0]
    j = rows * table.shape[1]
    tx = min(c // 4, 64)
    return _GATHER_ACT_KERNEL(
        inputs=[x, table.reshape(-1), mean, inv, weight.reshape(-1), bias.reshape(-1),
                mx.array([n, j], dtype=mx.int32)],
        template=[("T", dtype), ("X", x.dtype), ("C", c)],
        grid=(c // 4, j, 1),
        threadgroup=(tx, max(1, 256 // tx), 1),
        output_shapes=[(rows, table.shape[1] * c)],
        output_dtypes=[dtype],
    )[0]
