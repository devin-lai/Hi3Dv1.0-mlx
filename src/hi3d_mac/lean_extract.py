"""Lower-memory, bitwise-identical replacements for three steps of mesh extraction.

At 1024³ the official extraction (``point_to_mesh``: sparse cube vertices, FlexiCubes)
handles ~4 M voxels, 32 M cube corners and 40 M cube edges, and its row-unique and
scatter steps materialise ``[rows, 3]`` / ``[rows, 5]`` int64 index tensors several
times over. These replacements compute the same outputs from packed int64 keys:

* :func:`construct_voxel_grid`: corner ``(x, y, z)`` rows become one int64 key per
  corner (``[V, 8]`` instead of ``[8V, 3]``); keys preserve lexicographic order, so
  the sorted unique keys decode to exactly ``torch.unique(dim=0)``'s rows and inverse.
* :func:`cubes_to_verts`: the ``mean`` scatter runs one channel at a time with a 1-D
  index, instead of expanding the index to ``[8V, channels]`` int64. PyTorch's CPU
  scatter accumulates each output element in source order either way, so the sums
  and counts are the same.
* :func:`identify_surf_edges` (``FlexiCubes._identify_surf_edges``): edge
  ``(v0, v1)`` rows become keys ``v0 * n_vertices + v1``.

The sort runs on the GPU with MLX (:func:`hi3d_mac.fast_unique.unique_keys`), which also
serves the remaining ``torch.unique(dim=0)`` calls. ``tests/test_backends.py``
checks each function against the upstream original; the extracted meshes are
bitwise identical.
"""

from __future__ import annotations

import numpy as np
import torch

from hi3d_mac.fast_unique import unique_keys


# ----------------------------------------------------------------------------- sparse cubes

_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                    dtype=np.int64)
"""``sparse_cube.CUBE_CORNERS``: corner offsets in FlexiCubes' order."""


def construct_voxel_grid(coords: torch.Tensor):
    """``sparse_cube.construct_voxel_grid``: unique cube corners and per-cube indices."""
    c = coords.numpy().astype(np.int64, copy=False)
    if len(c) == 0:
        return coords.new_empty((0, 3)), torch.empty((0, 8), dtype=torch.int64)
    lo = c.min(axis=0)
    extent = c.max(axis=0) - lo + 2  # corners reach max + 1
    base = (c[:, 0] - lo[0]) * extent[1]
    base += c[:, 1] - lo[1]
    base *= extent[2]
    base += c[:, 2] - lo[2]
    offsets = (_CORNERS[:, 0] * extent[1] + _CORNERS[:, 1]) * extent[2] + _CORNERS[:, 2]
    keys = (base[:, None] + offsets[None, :]).reshape(-1)
    del base
    uniq, inverse, _ = unique_keys(keys)
    del keys
    verts = np.empty((uniq.shape[0], 3), dtype=np.int64)
    verts[:, 2] = uniq % extent[2] + lo[2]
    rest = uniq // extent[2]
    verts[:, 1] = rest % extent[1] + lo[1]
    verts[:, 0] = rest // extent[1] + lo[0]
    return torch.from_numpy(verts).to(coords.dtype), torch.from_numpy(inverse).reshape(-1, 8)


def cubes_to_verts(num_verts: int, cubes: torch.Tensor, value: torch.Tensor, reduce: str = "mean"):
    """``sparse_cube.cubes_to_verts`` with a 1-D index per channel."""
    index = cubes.reshape(-1)
    channels = value.shape[2]
    out = torch.empty(num_verts, channels, dtype=value.dtype)
    for ch in range(channels):
        src = value[:, :, ch].reshape(-1)
        reduced = torch.zeros(num_verts, dtype=value.dtype)
        out[:, ch] = torch.scatter_reduce(reduced, 0, index, src, reduce=reduce, include_self=False)
    return out


# ----------------------------------------------------------------------------- FlexiCubes


@torch.no_grad()
def identify_surf_edges(self, scalar_field, cube_idx, surf_cubes):
    """``FlexiCubes._identify_surf_edges`` with packed edge keys."""
    occ_n = scalar_field < 0
    num_verts = scalar_field.shape[0]
    corners = cube_idx[surf_cubes].numpy().astype(np.int64, copy=False)  # [S, 8]
    edges = self.cube_edges.numpy().astype(np.int64)
    keys = corners[:, edges[0::2]]  # [S, 12]: the pairs of cube_edges, in order
    keys *= num_verts
    keys += corners[:, edges[1::2]]
    del corners
    uniq, inverse, counts = unique_keys(keys.reshape(-1), return_counts=True)
    del keys
    unique_edges = torch.from_numpy(np.stack([uniq // num_verts, uniq % num_verts], axis=1).astype(np.int32))
    _idx_map = torch.from_numpy(inverse)
    counts = torch.from_numpy(counts)
    # From here on the upstream body, unchanged.
    mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

    surf_edges_mask = mask_edges[_idx_map]
    counts = counts[_idx_map]

    mapping = torch.ones((unique_edges.shape[0]), dtype=torch.int32, device=cube_idx.device) * -1
    mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.int32, device=cube_idx.device)
    idx_map = mapping[_idx_map]
    surf_edges = unique_edges[mask_edges]
    return surf_edges, idx_map, counts, surf_edges_mask


def install() -> None:
    """Swap the three upstream functions (CPU tensors only, as the pipeline uses them)."""
    from sparc3d.ops.flexicubes.flexicubes import FlexiCubes
    from sparc3d.sparcubes import sparse_cube

    sparse_cube.construct_voxel_grid = construct_voxel_grid
    sparse_cube.cubes_to_verts = cubes_to_verts
    FlexiCubes._identify_surf_edges = identify_surf_edges
