"""NumPy port of ``sparc3d/ops/triton/voxelize.py`` and a device-neutral ``adaptive_vox``.

The Triton kernels test every voxel in each triangle's bounding box with the
separating-axis triangle/box overlap test (box half-size 0.5 around the voxel
centre). This module evaluates the same float32 expressions, in the same order,
vectorised over candidate (triangle, voxel) pairs, then sorts the hits by the
same ``z << 22 | y << 11 | x`` key. ``adaptive_vox`` and ``MeshVoxelizer`` are the
official functions with their CUDA-device checks removed.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

CANDIDATES_PER_CHUNK = 2_000_000
"""Candidate (triangle, voxel) pairs tested per vectorised batch."""

DEFAULT_PAD = 3
MIN_RESOLUTION = 2 * DEFAULT_PAD + 1
MAX_RESOLUTION = 2048

_HALF = np.float32(0.5)


def _axis_overlaps(v0, v1, v2, axis) -> np.ndarray:
    ax, ay, az = axis
    p0 = v0[0] * ax + v0[1] * ay + v0[2] * az
    p1 = v1[0] * ax + v1[1] * ay + v1[2] * az
    p2 = v2[0] * ax + v2[1] * ay + v2[2] * az
    lower = np.minimum(np.minimum(p0, p1), p2)
    upper = np.maximum(np.maximum(p0, p1), p2)
    radius = _HALF * (np.abs(ax) + np.abs(ay) + np.abs(az))
    return (lower <= radius) & (upper >= -radius)


def _triangle_overlaps_voxel(a, b, c, x, y, z) -> np.ndarray:
    """``_triangle_overlaps_voxel`` from the Triton kernel; all arrays float32."""
    cx = x.astype(np.float32) + _HALF
    cy = y.astype(np.float32) + _HALF
    cz = z.astype(np.float32) + _HALF
    v0 = (a[0] - cx, a[1] - cy, a[2] - cz)
    v1 = (b[0] - cx, b[1] - cy, b[2] - cz)
    v2 = (c[0] - cx, c[1] - cy, c[2] - cz)
    overlaps = np.ones(cx.shape, dtype=bool)
    for i in range(3):
        overlaps &= np.minimum(np.minimum(v0[i], v1[i]), v2[i]) <= _HALF
        overlaps &= np.maximum(np.maximum(v0[i], v1[i]), v2[i]) >= -_HALF
    e0 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    e1 = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
    e2 = (v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2])
    zero = np.float32(0.0)
    for e in (e0, e1, e2):
        overlaps &= _axis_overlaps(v0, v1, v2, (zero, e[2], -e[1]))
        overlaps &= _axis_overlaps(v0, v1, v2, (-e[2], zero, e[0]))
        overlaps &= _axis_overlaps(v0, v1, v2, (e[1], -e[0], zero))
    nx = e0[1] * e1[2] - e0[2] * e1[1]
    ny = e0[2] * e1[0] - e0[0] * e1[2]
    nz = e0[0] * e1[1] - e0[1] * e1[0]
    plane_offset = -(nx * v0[0] + ny * v0[1] + nz * v0[2])
    plane_radius = _HALF * (np.abs(nx) + np.abs(ny) + np.abs(nz))
    return overlaps & (np.abs(plane_offset) <= plane_radius)


def coarse_voxelize_numpy(
    vertices: np.ndarray, faces: np.ndarray, resolution: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Surface-voxelize triangles given in voxel space.

    Returns:
        ``(keys, face_ids)`` of every (voxel, triangle) hit, in emission order.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    tri = vertices[faces]  # [F, 3 corners, 3 axes]
    lo = np.maximum(np.floor(tri.min(axis=1)).astype(np.int32), 0)
    hi = np.minimum(np.floor(tri.max(axis=1)).astype(np.int32), resolution - 1)
    extent = np.maximum(hi - lo + 1, 0).astype(np.int64)
    counts = extent.prod(axis=1)

    keys_out, faces_out = [], []
    face_start = 0
    cumulative = np.cumsum(counts)
    while face_start < faces.shape[0]:
        base = cumulative[face_start - 1] if face_start else 0
        face_stop = int(np.searchsorted(cumulative, base + CANDIDATES_PER_CHUNK, "right"))
        face_stop = max(face_stop, face_start + 1)
        sel = np.arange(face_start, face_stop)
        n = counts[sel]
        total = int(n.sum())
        face_start = face_stop
        if total == 0:
            continue
        fid = np.repeat(sel, n)
        local = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
        ex, ey = extent[fid, 0], extent[fid, 1]
        x = lo[fid, 0] + (local % ex)
        y = lo[fid, 1] + (local // ex) % ey
        z = lo[fid, 2] + local // (ex * ey)
        t = tri[fid]
        hit = _triangle_overlaps_voxel(t[:, 0].T, t[:, 1].T, t[:, 2].T, x, y, z)
        # Triton emits hits per face in z, y, x loop order; x varies fastest here too.
        keys_out.append((z[hit] << 22) | (y[hit] << 11) | x[hit])
        faces_out.append(fid[hit])
    if not keys_out:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(keys_out), np.concatenate(faces_out)


class _NumpyVoxelOps:
    """Stand-in for ``sparc3d.ops.triton`` returning the same CSR tuple."""

    @staticmethod
    def coarse_voxelize(vertices: torch.Tensor, faces: torch.Tensor, resolution: int):
        if resolution < 1 or resolution > 2048:
            raise ValueError(f"resolution must be in [1, 2048], got {resolution}")
        device = vertices.device
        keys, face_ids = coarse_voxelize_numpy(
            vertices.detach().to("cpu", torch.float32).numpy(),
            faces.detach().to("cpu", torch.int64).numpy(),
            resolution,
        )
        num_faces = faces.shape[0]
        face_voxel_count = np.bincount(face_ids, minlength=num_faces).astype(np.int32)
        order = np.argsort(keys, kind="stable")
        unique_keys, unique_counts = np.unique(keys[order], return_counts=True)
        starts = np.concatenate([[0], np.cumsum(unique_counts)[:-1]]).astype(np.int32)
        as_t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)
        return (
            as_t(unique_keys, torch.int64),
            as_t(face_ids[order], torch.int64),
            as_t(starts, torch.int32),
            as_t(unique_counts, torch.int32),
            as_t(face_voxel_count, torch.int32),
        )


def install(voxelize_module) -> None:
    """Point the official voxelization module at the NumPy kernels."""
    official_cls = voxelize_module.MeshVoxelizer

    class MeshVoxelizer(official_cls):
        """Official ``MeshVoxelizer`` without its CUDA-tensor checks."""

        def __init__(
            self,
            vertices: torch.Tensor,
            faces: torch.Tensor,
            resolution: int,
            eps: float = 1e-6,
            *,
            pad: int = DEFAULT_PAD,
            shift: Optional[Union[torch.Tensor, Sequence[float], float]] = None,
        ):
            self.resolution = int(resolution)
            self.pad = int(pad)
            self.eps = float(eps)
            if not 1 <= self.resolution <= MAX_RESOLUTION:
                raise ValueError(f"resolution must be in [1, {MAX_RESOLUTION}]")
            self.faces = faces
            self.original_vertices = vertices
            self.vertices_min = self.original_vertices.min(dim=0)[0]
            self.vertices_max = self.original_vertices.max(dim=0)[0]
            self.expanded_min = self.vertices_min - self.eps
            self.expanded_max = self.vertices_max + self.eps
            self.center = (self.expanded_min + self.expanded_max) / 2.0
            max_range = (self.expanded_max - self.expanded_min).max()
            self.scale = (self.resolution - self.pad * 2 - 2 * self.eps) / max_range
            self.shift = self._normalize_shift(shift)
            self.norm_vertices = self.vertex_to_voxel()

    def adaptive_vox(mesh, resolution: int = 64, target_num: int = 80000, device="cpu"):
        """Official ``adaptive_vox`` (shrink-to-fit voxelization), CPU tensors."""
        import trimesh

        if target_num < 1:
            raise ValueError(f"target_num must be positive, got {target_num}")
        if resolution < MIN_RESOLUTION:
            raise ValueError(f"resolution must be at least {MIN_RESOLUTION}")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.int32)
        scale = 1.0
        while True:
            cur_res = int(resolution * scale)
            if cur_res < MIN_RESOLUTION:
                raise ValueError(
                    f"shrinking from resolution {resolution} reached {cur_res}; "
                    f"target_num={target_num} is unreachable for this mesh"
                )
            vox = MeshVoxelizer(vertices, faces, resolution=cur_res)
            coords = vox.voxelize()["voxel_coords"].to(dtype=torch.int32)
            if coords.shape[0] <= target_num:
                break
            scale = scale * 0.95
        if cur_res != resolution:
            warnings.warn(
                f"adaptive_vox settled at resolution {cur_res} instead of the requested "
                f"{resolution} to fit target_num={target_num}",
                stacklevel=2,
            )
        return coords.to(device)

    voxelize_module.load_triton_ops = lambda: _NumpyVoxelOps
    voxelize_module.MeshVoxelizer = MeshVoxelizer
    voxelize_module.adaptive_vox = adaptive_vox
