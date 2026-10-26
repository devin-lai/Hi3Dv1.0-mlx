"""``flash_attn`` varlen functions implemented with exact (non-approximate) attention.

flash-attn computes ``softmax(q k^T * scale) v`` per packed sequence with fp32
accumulation. Here every sequence is evaluated with
``torch.nn.functional.scaled_dot_product_attention`` in float32, split into query
blocks so the score matrix never exceeds :data:`SCORE_BYTES`. Splitting queries
leaves each softmax row intact, so the result is the same attention, not an
approximation. Output is returned in the input dtype, as flash-attn does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SCORE_BYTES = 768 * 1024 * 1024
"""Largest per-block fp32 score matrix ``heads x q_block x keys`` materialized."""


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Batched attention ``[B, L, H, D]`` -> ``[B, L, H, D]`` in float32."""
    batch, q_len, heads, _ = q.shape
    k_len = k.shape[1]
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2)
    vf = v.float().transpose(1, 2)
    per_query = batch * heads * k_len * 4
    block = max(1, min(q_len, SCORE_BYTES // max(1, per_query)))
    if block >= q_len:
        out = F.scaled_dot_product_attention(qf, kf, vf, scale=scale)
    else:
        out = torch.cat(
            [
                F.scaled_dot_product_attention(qf[:, :, s : s + block], kf, vf, scale=scale)
                for s in range(0, q_len, block)
            ],
            dim=2,
        )
    return out.transpose(1, 2)


def _varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    bounds_q = cu_q.tolist()
    bounds_k = cu_k.tolist()
    lens_q = [b - a for a, b in zip(bounds_q, bounds_q[1:])]
    lens_k = [b - a for a, b in zip(bounds_k, bounds_k[1:])]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    if len(set(lens_q)) == 1 and len(set(lens_k)) == 1:
        # Uniform windows (serialized VAE attention) or a single sequence.
        n, lq, lk = len(lens_q), lens_q[0], lens_k[0]
        qb = q[: n * lq].reshape(n, lq, *q.shape[1:])
        kb = k[: n * lk].reshape(n, lk, *k.shape[1:])
        vb = v[: n * lk].reshape(n, lk, *v.shape[1:])
        per_window = lq * lk * q.shape[1] * 4
        windows = max(1, SCORE_BYTES // max(1, per_window))
        for s in range(0, n, windows):
            out[s * lq : (s + windows) * lq] = _attend(
                qb[s : s + windows], kb[s : s + windows], vb[s : s + windows], scale
            ).reshape(-1, *q.shape[1:])
    else:
        for i in range(len(lens_q)):
            qs = slice(bounds_q[i], bounds_q[i + 1])
            ks = slice(bounds_k[i], bounds_k[i + 1])
            out[qs] = _attend(q[qs][None], k[ks][None], v[ks][None], scale)[0]
    return out.to(q.dtype)


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_: object,
) -> torch.Tensor:
    """Self-attention over packed ``[T, 3, H, D]`` sequences."""
    if causal or dropout_p:
        raise NotImplementedError("causal or dropout attention is not used by Hi3D")
    q, k, v = qkv.unbind(1)
    return _varlen(q, k, v, cu_seqlens, cu_seqlens, softmax_scale)


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_: object,
) -> torch.Tensor:
    """Cross-attention of packed queries ``[T, H, D]`` over packed ``[S, 2, H, D]``."""
    if causal or dropout_p:
        raise NotImplementedError("causal or dropout attention is not used by Hi3D")
    k, v = kv.unbind(1)
    return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
