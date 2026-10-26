"""Exact drop-in for ``torch.unique(x, dim=0, ...)`` on integer rows, on the GPU.

Mesh extraction calls ``torch.unique(..., dim=0)`` on tens of millions of integer
rows (cube corners, cube edges, boundary edges); PyTorch's CPU ``unique_dim`` is
the slowest step of the pipeline. Packing each row into one int64 key preserves
lexicographic order for non-negative offsets, so sorting the keys with MLX and
marking run boundaries yields the same unique rows (sorted), inverse indices and
counts as ``torch.unique(dim=0)``.

:func:`install` swaps ``torch`` in the three upstream extraction modules for a proxy
whose ``unique`` takes this path; the upstream source files are unchanged.
"""

from __future__ import annotations

import types

import mlx.core as mx
import numpy as np
import torch

_INT_DTYPES = (torch.int16, torch.int32, torch.int64, torch.uint8)


def unique_keys(keys: np.ndarray, return_counts: bool = False):
    """Sorted unique values of a 1-D int64 array, the inverse indices and counts.

    Same results as ``np.unique(keys, return_inverse=True, return_counts=...)``. The
    sort runs on the GPU; the inverse is scattered there too, so only the keys, the
    unique values, the inverse and the counts ever exist on the CPU.
    """
    n = keys.shape[0]
    if n == 0:
        return keys.copy(), np.empty(0, dtype=np.int64), (np.empty(0, dtype=np.int64) if return_counts else None)
    k = mx.array(keys)
    order = mx.argsort(k)
    sorted_keys = mx.take(k, order)
    del k
    starts = mx.concatenate([mx.array([True]), sorted_keys[1:] != sorted_keys[:-1]])
    run = mx.cumsum(starts.astype(mx.int32)) - 1
    inverse_mx = mx.zeros((n,), dtype=mx.int32).at[order].add(run)
    mx.eval(sorted_keys, starts, inverse_mx)
    del order, run
    starts_np = np.array(starts)
    uniq = np.array(sorted_keys)[starts_np]
    inverse = np.array(inverse_mx).astype(np.int64)
    del sorted_keys, starts, inverse_mx
    counts = None
    if return_counts:
        boundaries = np.flatnonzero(starts_np)
        counts = np.diff(np.append(boundaries, n)).astype(np.int64)
    # The sort buffers (tens of millions of keys at 1024³) are one-off: hand them
    # back to the OS instead of keeping them in MLX's buffer cache.
    mx.clear_cache()
    return uniq, inverse, counts


def _packed_keys(a: np.ndarray):
    """One int64 key per row, in the rows' lexicographic order (``None`` if too wide)."""
    lo = a.min(axis=0)
    extent = a.max(axis=0) - lo + 1
    if float(np.prod(extent.astype(np.float64))) >= 2**62:
        return None
    key = a[:, 0] - lo[0]
    for j in range(1, a.shape[1]):
        key *= int(extent[j])
        key += a[:, j]
        key -= lo[j]
    return key, lo, extent


def unique_rows(x: torch.Tensor, return_inverse: bool = False, return_counts: bool = False):
    a = x.numpy().astype(np.int64, copy=False)
    packed = _packed_keys(a) if a.shape[0] else None
    if packed is None:
        return _torch_unique(x, dim=0, return_inverse=return_inverse, return_counts=return_counts)
    key, lo, extent = packed
    uniq, inverse, counts = unique_keys(key, return_counts=return_counts)
    del key
    rows = np.empty((uniq.shape[0], a.shape[1]), dtype=np.int64)
    rem = uniq
    for j in range(a.shape[1] - 1, 0, -1):
        rows[:, j] = rem % extent[j] + lo[j]
        rem = rem // extent[j]
    rows[:, 0] = rem + lo[0]
    out = [torch.from_numpy(rows).to(x.dtype)]
    if return_inverse:
        out.append(torch.from_numpy(inverse))
    if return_counts:
        out.append(torch.from_numpy(counts))
    return out[0] if len(out) == 1 else tuple(out)


_torch_unique = torch.unique


def _unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if (
        dim == 0
        and input.device.type == "cpu"
        and input.dtype in _INT_DTYPES
        and input.ndim == 2
        and sorted
    ):
        return unique_rows(input, return_inverse, return_counts)
    return _torch_unique(input, sorted=sorted, return_inverse=return_inverse,
                         return_counts=return_counts, dim=dim)


class _TorchProxy(types.ModuleType):
    """``torch`` with ``unique`` replaced; everything else delegates to torch."""

    def __init__(self):
        super().__init__("torch")

    def __getattr__(self, name):
        return getattr(torch, name)

    unique = staticmethod(_unique)


def install() -> None:
    from sparc3d.ops.flexicubes import flexicubes
    from sparc3d.sparcubes import fill_hole, sparse_cube

    proxy = _TorchProxy()
    for module in (flexicubes, fill_hole, sparse_cube):
        module.torch = proxy
