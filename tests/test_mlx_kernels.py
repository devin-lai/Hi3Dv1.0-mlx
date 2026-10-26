"""Checks that the MLX restructurings and Metal kernels keep the reference arithmetic
(and that the flash-attention kernel is as accurate as fp16 output allows).

Run: ``.venv/bin/python -m pytest mac/tests -q`` from ``Hi3Dv1.0``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MLX_ENABLE_TF32", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

mx = pytest.importorskip("mlx.core")

from hi3d_mac.mlx_backend import kernels, sparse  # noqa: E402
from hi3d_mac.mlx_backend.transformer import apply_rope, layer_norm32, rms_norm_heads  # noqa: E402

F16, F32 = mx.float16, mx.float32


@pytest.mark.parametrize("dtype", [F16, F32])
def test_qkv_rope_rms_matches_composed_ops(dtype):
    b, n, h, d = 2, 37, 3, 96
    x = (mx.random.normal((b * n, 3 * h * d), key=mx.random.key(0)) * 2).astype(dtype)
    cos = mx.random.uniform(-1, 1, (n, d // 2), key=mx.random.key(1))
    sin = mx.random.uniform(-1, 1, (n, d // 2), key=mx.random.key(2))
    gq = mx.random.uniform(0.5, 1.5, (h, d), key=mx.random.key(3))
    gk = mx.random.uniform(0.5, 1.5, (h, d), key=mx.random.key(4))
    r = x.reshape(b * n, 3, h, d)
    cb, sb = mx.tile(cos, (b, 1)), mx.tile(sin, (b, 1))
    to = lambda t: t.reshape(b, n, h, d).transpose(0, 2, 1, 3)
    ref = (to(rms_norm_heads(apply_rope(r[:, 0], cb, sb), gq, dtype)),
           to(rms_norm_heads(apply_rope(r[:, 1], cb, sb), gk, dtype)), to(r[:, 2]))
    got = kernels.qkv_rope_rms(x, cos, sin, gq, gk, b, h, d)
    tol = 8e-3 if dtype == F16 else 1e-5  # one fp16 ulp at |x| ~ 4; fp32 summation order
    for a, c in zip(ref, got):
        assert a.shape == c.shape
        assert mx.allclose(a.astype(F32), c.astype(F32), atol=tol, rtol=tol).item()
    assert mx.array_equal(ref[2], got[2]).item()  # values are copied, not recomputed


def test_layer_norm_modulate_matches_composed_ops():
    x = (mx.random.normal((2, 50, 2112), key=mx.random.key(5)) * 3 + 0.5).astype(F16)
    scale = (1 + 0.1 * mx.random.normal((1, 2112), key=mx.random.key(6))).astype(F16)
    shift = (0.1 * mx.random.normal((1, 2112), key=mx.random.key(7))).astype(F16)
    w, bias = mx.random.normal((2112,), key=mx.random.key(8)), mx.random.normal((2112,), key=mx.random.key(9))
    for ref, got in (
        (layer_norm32(x) * scale + shift, kernels.layer_norm_modulate(x, 1e-6, scale=scale, shift=shift)),
        (layer_norm32(x, w, bias), kernels.layer_norm_modulate(x, 1e-6, weight=w, bias=bias)),
        (layer_norm32(x), kernels.layer_norm_modulate(x, 1e-6)),
    ):
        equal = (ref == got).mean().item()
        assert equal > 0.999, equal  # the rest differ by one ulp (fp32 summation order)
        assert mx.allclose(ref.astype(F32), got.astype(F32), atol=1e-2, rtol=1e-3).item()


@pytest.mark.parametrize("dtype", [F32, F16])
def test_gather_group_norm_swish_matches_composed_ops(dtype):
    n, c = 300, 64
    x = mx.random.normal((n, c), key=mx.random.key(10))
    mean = mx.random.normal((c,), key=mx.random.key(11))
    inv = mx.random.uniform(0.5, 2.0, (c,), key=mx.random.key(12))
    w, b = mx.random.normal((1, c), key=mx.random.key(13)), mx.random.normal((1, c), key=mx.random.key(14))
    table = mx.array(np.random.default_rng(0).integers(0, n + 1, (120, 27)).astype(np.int32))  # n = absent
    a = (x - mean) * inv * w + b
    a = a * mx.sigmoid(a)
    ref = mx.take(sparse.pad_rows(a), table, axis=0).reshape(120, 27 * c).astype(dtype)
    got = kernels.gather_group_norm_swish(x, table, mean, inv, w, b, dtype)
    tol = 2e-3 if dtype == F16 else 1e-5
    assert mx.allclose(ref.astype(F32), got.astype(F32), atol=tol, rtol=tol).item()


def test_child_neighbour_table_matches_coordinate_hashing():
    rng = np.random.default_rng(1)
    occ = rng.random((10, 10, 10)) < 0.35
    coords = np.argwhere(occ).astype(np.int32)
    level = sparse.Level.from_coords(coords)
    mask = rng.random((len(coords), 8)) < 0.5
    table, parent, slot = sparse.child_neighbour_table(level.table, mask)
    bits = np.stack([(slot >> 2) & 1, (slot >> 1) & 1, slot & 1], 1)
    child_coords = coords[parent] * 2 + bits
    expected = sparse.neighbour_table(child_coords)  # dense-grid lookup of the children themselves
    assert np.array_equal(np.array(table), expected)


def test_blocked_assembly_matches_single_block():
    x = mx.random.normal((1000, 16), key=mx.random.key(15))
    fn = lambda s, e: x[s:e] * 2 + 1
    whole = fn(0, 1000)
    assert mx.array_equal(sparse.blocked(1000, 64, fn, 16, F32), whole).item()
    old = sparse.ASSEMBLE_IN_PLACE_BYTES
    try:
        sparse.ASSEMBLE_IN_PLACE_BYTES = 0  # force the in-place path
        assert mx.array_equal(sparse.blocked(1000, 64, fn, 16, F32), whole).item()
    finally:
        sparse.ASSEMBLE_IN_PLACE_BYTES = old


def test_blocked_accumulate_adds_in_place():
    n, c = 200_000, 128
    base = mx.random.normal((n, c), key=mx.random.key(16))
    h = mx.random.normal((n, c), key=mx.random.key(17))
    mx.eval(base, h)
    expected = np.array(base + h * 2)
    fn = lambda s, e: h[s:e] * 2
    holder = [base]
    del base  # donation requires sole ownership BEFORE calling blocked
    mx.synchronize()
    mx.reset_peak_memory()
    before = mx.get_active_memory()
    out = sparse.blocked(n, 20_000, fn, c, F32, accumulate=holder)
    mx.eval(out)
    assert np.array_equal(np.array(out), expected)
    # no second full-size array: the peak stays within one block of the inputs
    assert mx.get_peak_memory() - before < n * c * 4 // 2


def test_vae_residual_in_place_matches_composed(monkeypatch):
    from hi3d_mac.mlx_backend import vae as vae_module

    rng = np.random.default_rng(3)
    side = 16
    idx = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3)
    coords = idx[rng.random(idx.shape[0]) < 0.5].astype(np.int32)
    level = sparse.Level.from_coords(coords)
    c = 32
    p = {}
    for name in ("r.block.0", "r.block.3"):
        p[f"{name}.weights"] = mx.array(rng.uniform(0.5, 1.5, (1, c)).astype(np.float32))
        p[f"{name}.bias"] = mx.array(rng.normal(0, 0.1, (1, c)).astype(np.float32))
    for name in ("r.block.2", "r.block.5"):
        p[f"{name}.weight"] = mx.array((rng.normal(0, 0.05, (27 * c, c))).astype(np.float32))
        p[f"{name}.bias"] = mx.array(rng.normal(0, 0.1, (c,)).astype(np.float32))
    dec = vae_module.SparconvVAEDecoderMLX.__new__(vae_module.SparconvVAEDecoderMLX)
    dec.dt, dec.act, dec.p = F32, F32, p
    x = mx.array(rng.normal(0, 1, (coords.shape[0], c)).astype(np.float32))
    # Keep GEMM shapes identical: changing row blocks also changes the GEMM's
    # accumulation order, which is unrelated to residual buffer donation.
    monkeypatch.setattr(sparse, "GATHER_ELEMENTS", 27 * c * 97)
    monkeypatch.setattr(vae_module, "EVAL_POINTS", 10**9)  # composed path
    composed = np.array(dec.resnet([x], level, "r"))
    monkeypatch.setattr(vae_module, "EVAL_POINTS", 0)  # in-place path
    in_place = np.array(dec.resnet([mx.array(np.array(x))], level, "r"))
    assert np.array_equal(composed, in_place)


def _exact_attention(q, k, v, scale):
    qf, kf, vf = (np.asarray(t.astype(mx.float32), np.float64) for t in (q, k, v))
    kf = np.broadcast_to(kf, (qf.shape[0],) + kf.shape[1:])
    vf = np.broadcast_to(vf, (qf.shape[0],) + vf.shape[1:])
    s = np.einsum("bhnd,bhmd->bhnm", qf, kf) * scale
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bhnm,bhmd->bhnd", p, vf).transpose(0, 2, 1, 3).reshape(qf.shape[0], qf.shape[2], -1)


@pytest.mark.parametrize("batch,nq,nk,shared", [(2, 333, 333, False), (1, 77, 291, True), (2, 300, 129, True),
                                                (1, 5, 3, False), (2, 1000, 31, False)])
def test_flash_attention_matches_float64(batch, nq, nk, shared):
    from hi3d_mac.mlx_backend.attention import flash_attention

    rng = np.random.default_rng(0)
    h, d = 22, 96
    q = mx.array((rng.standard_normal((batch, h, nq, d)) * 2).astype(np.float16))
    k = mx.array((rng.standard_normal((1 if shared else batch, h, nk, d)) * 2).astype(np.float16))
    v = mx.array(rng.standard_normal(k.shape).astype(np.float16))
    exact = _exact_attention(q, k, v, d**-0.5)
    out = np.asarray(flash_attention(q, k, v, d**-0.5).astype(mx.float32), np.float64)
    err = np.sqrt(((out - exact) ** 2).mean() / (exact**2).mean())
    assert err < 3e-4, err  # fp16 output rounding is ~2e-4
    # MLX's own kernel, for reference, lands at ~2.4e-4 on these inputs
    bnhd = flash_attention(q.transpose(0, 2, 1, 3), k, v, d**-0.5, q_layout="bnhd")
    assert np.array_equal(np.asarray(bnhd.astype(mx.float32), np.float64), out)
