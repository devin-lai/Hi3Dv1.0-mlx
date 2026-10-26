"""Checks that the Mac replacements for spconv / flash-attn / Triton are exact.

Run: ``.venv/bin/python -m pytest mac/tests -q`` from ``Hi3Dv1.0``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hi3d_mac.torch_backend import attention, sparse_conv, voxelize  # noqa: E402


def _random_sparse(n_side=14, density=0.15, channels=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    occ = torch.rand(n_side, n_side, n_side, generator=g) < density
    xyz = occ.nonzero().int()
    perm = torch.randperm(xyz.shape[0], generator=g)
    xyz = xyz[perm]  # arbitrary row order, as in real use
    feats = torch.randn(xyz.shape[0], channels, generator=g, dtype=torch.float64)
    indices = torch.cat([torch.zeros(xyz.shape[0], 1, dtype=torch.int32), xyz], 1)
    return indices, feats


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_subm_conv_matches_dense_conv3d(device):
    indices, feats = _random_sparse()
    n_side = 14
    conv = sparse_conv.SubMConv3d(5, 7, 3, bias=True, indice_key="subm3").double()
    torch.nn.init.normal_(conv.bias)
    x = sparse_conv.SparseConvTensor(feats, indices, [n_side] * 3, 1)
    dtype = torch.float32 if device == "mps" else torch.float64
    conv_d = conv.to(device=device, dtype=dtype)
    xd = sparse_conv.SparseConvTensor(feats.to(device, dtype), indices.to(device), [n_side] * 3, 1)
    out = conv_d(xd).features.cpu().double()

    dense = torch.zeros(1, 5, n_side, n_side, n_side, dtype=torch.float64)
    b, i, j, k = indices.long().unbind(1)
    dense[0, :, i, j, k] = feats.T
    w = conv.weight.detach().cpu().double().permute(0, 4, 1, 2, 3)  # (out, in, kx, ky, kz)
    ref = F.conv3d(dense, w, conv.bias.detach().cpu().double(), padding=1)[0, :, i, j, k].T
    tol = 1e-5 if device == "mps" else 1e-12
    assert torch.allclose(out, ref, atol=tol, rtol=tol), (out - ref).abs().max()


def test_pair_table_matches_spconv_layout():
    indices, _ = _random_sparse(n_side=8, density=0.4)
    out_inds, pairs, counts = sparse_conv.get_indice_pairs(
        indices, 1, [8, 8, 8], sparse_conv.ConvAlgo.Native, (3, 3, 3), (1, 1, 1),
        (1, 1, 1), (1, 1, 1), (0, 0, 0), True, False,
    )
    coords = indices[:, 1:].long()
    for k in range(27):
        r = torch.tensor([k // 9, (k // 3) % 3, k % 3])
        n = int(counts[k])
        src, dst = pairs[0, k, :n], pairs[1, k, :n]
        # spconv: an input at p feeds the output at p + center - r.
        assert torch.equal(coords[dst], coords[src] + 1 - r)
        # Submanifold pairs are symmetric under k -> 26 - k.
        assert n == int(counts[26 - k])


def _naive_attention(q, k, v, scale):
    s = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * scale
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), v.double())


def test_varlen_attention_single_and_windows():
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(96, 4, 16, generator=g) for _ in range(3))
    cu = torch.tensor([0, 96], dtype=torch.int32)
    out = attention.flash_attn_varlen_qkvpacked_func(torch.stack([q, k, v], 1), cu, 96)
    assert torch.allclose(out.double(), _naive_attention(q, k, v, 16**-0.5), atol=1e-5)
    cu = torch.arange(0, 97, 48, dtype=torch.int32)
    out = attention.flash_attn_varlen_qkvpacked_func(
        torch.stack([q, k, v], 1), cu, 48, softmax_scale=0.3
    )
    for s in range(0, 96, 48):
        ref = _naive_attention(q[s : s + 48], k[s : s + 48], v[s : s + 48], 0.3)
        assert torch.allclose(out[s : s + 48].double(), ref, atol=1e-5)


def test_cross_attention_matches_naive():
    g = torch.Generator().manual_seed(1)
    q = torch.randn(50, 2, 8, generator=g)
    kv = torch.randn(30, 2, 2, 8, generator=g)
    out = attention.flash_attn_varlen_kvpacked_func(
        q, kv, torch.tensor([0, 50]), torch.tensor([0, 30]), 50, 30
    )
    ref = _naive_attention(q, kv[:, 0], kv[:, 1], 8**-0.5)
    assert torch.allclose(out.double(), ref, atol=1e-5)


def _brute_force_voxels(vertices, faces, res):
    """Every voxel whose box overlaps a triangle, in float64."""
    hits = set()
    for tri in vertices[faces].astype(np.float64):
        lo = np.maximum(np.floor(tri.min(0)).astype(int), 0)
        hi = np.minimum(np.floor(tri.max(0)).astype(int), res - 1)
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    t = tri - (np.array([x, y, z]) + 0.5)
                    if _sat(t):
                        hits.add((x, y, z))
    return hits


def _sat(t):
    v0, v1, v2 = t
    if np.any(t.min(0) > 0.5) or np.any(t.max(0) < -0.5):
        return False
    edges = (v1 - v0, v2 - v1, v0 - v2)
    axes = [np.cross(e, a) for e in edges for a in np.eye(3)]
    axes.append(np.cross(edges[0], edges[1]))
    for ax in axes:
        p = t @ ax
        r = 0.5 * np.abs(ax).sum()
        if p.min() > r or p.max() < -r:
            return False
    return True


def test_voxelizer_matches_float64_sat():
    rng = np.random.default_rng(0)
    res = 24
    vertices = rng.uniform(2, res - 2, size=(60, 3)).astype(np.float32)
    faces = rng.integers(0, 60, size=(40, 3))
    keys, _ = voxelize.coarse_voxelize_numpy(vertices, faces, res)
    keys = np.unique(keys)
    got = {(int(k & 2047), int((k >> 11) & 2047), int(k >> 22)) for k in keys}
    assert got == _brute_force_voxels(vertices, faces, res)


def _decoded_like(n_side=40, seed=0):
    """A closed, bumpy shell of voxels with per-corner SDF / deformation channels."""
    g = np.random.default_rng(seed)
    idx = np.stack(np.meshgrid(*[np.arange(n_side)] * 3, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(idx - n_side / 2 + 0.5, axis=1)
    coords = idx[np.abs(r - n_side / 3) < 2.5]
    coords = coords[g.permutation(coords.shape[0])]
    n = coords.shape[0]
    feat = np.tanh(g.standard_normal((n, 32)).astype(np.float32))
    sign = g.standard_normal((n, 8)).astype(np.float32)
    return coords.astype(np.int32), feat, sign


def test_lean_extraction_is_bitwise_identical():
    from hi3d_mac import lean_extract
    from hi3d_mac.torch_backend import install

    install()
    from sparc3d.ops.flexicubes.flexicubes import FlexiCubes
    from sparc3d.sparcubes import sparse_cube

    coords, feat, sign = _decoded_like()
    c = torch.from_numpy(coords.astype(np.int64))
    ref_v, ref_cubes = sparse_cube.construct_voxel_grid(c)
    new_v, new_cubes = lean_extract.construct_voxel_grid(c)
    assert torch.equal(ref_v, new_v) and torch.equal(ref_cubes, new_cubes)

    value = torch.from_numpy(np.random.default_rng(1).standard_normal((coords.shape[0], 8, 5)).astype(np.float32))
    ref = sparse_cube.cubes_to_verts(ref_v.shape[0], ref_cubes, value)
    new = lean_extract.cubes_to_verts(ref_v.shape[0], ref_cubes, value)
    assert torch.equal(ref, new)

    fc = FlexiCubes(device="cpu")
    scalar = torch.from_numpy(np.random.default_rng(2).standard_normal(ref_v.shape[0]).astype(np.float32))
    surf, _ = fc._identify_surf_cubes(scalar, ref_cubes)
    for a, b in zip(fc._identify_surf_edges(scalar, ref_cubes, surf),
                    lean_extract.identify_surf_edges(fc, scalar, ref_cubes, surf)):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_fast_unique_matches_torch_unique():
    from hi3d_mac import fast_unique

    g = torch.Generator().manual_seed(0)
    rows = torch.randint(-5, 40, (20000, 3), generator=g)
    ref = torch.unique(rows, dim=0, return_inverse=True, return_counts=True)
    new = fast_unique.unique_rows(rows, return_inverse=True, return_counts=True)
    for a, b in zip(ref, new):
        assert a.dtype == b.dtype and torch.equal(a, b)
