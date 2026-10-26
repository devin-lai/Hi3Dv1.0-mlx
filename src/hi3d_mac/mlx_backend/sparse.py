"""Sparse voxel structure for MLX: neighbour tables, resampling, convolution.

Coordinates are fixed for a whole sampling pass (and per decoder level), so every
index structure is built once and reused by all model evaluations: with NumPy for
the voxels a pass starts from, on the GPU for the decoder's octree children.

Convolution convention (verified against spconv in ``tests/test_backends.py``):
``out[o] = b + sum_k W[:, k, :] @ in[o + r_k - 1]`` with ``r_k`` the row-major
``(x, y, z)`` kernel position of flat offset ``k``.
"""

from __future__ import annotations

import itertools
import time

import mlx.core as mx
import numpy as np

OFFSETS = np.array(list(itertools.product(range(3), repeat=3)), dtype=np.int64) - 1
"""``[27, 3]`` neighbour displacement ``r_k - 1`` for each flat kernel offset."""

CENTER = 13

GATHER_ELEMENTS = 1 << 27
"""Elements per gathered ``[rows, 27, C]`` block (256 MB at fp16)."""


def neighbour_table(coords: np.ndarray) -> np.ndarray:
    """``[N, 27]`` row of each voxel's neighbours, ``N`` where absent.

    Uses a padded dense index grid; coordinates are at most 1535 per axis only at
    decoder levels, which use :func:`child_neighbour_table` instead.
    """
    coords = np.asarray(coords, dtype=np.int64)
    n = coords.shape[0]
    extent = coords.max(axis=0) + 3
    grid = np.full(tuple(extent), n, dtype=np.int32)
    grid[tuple((coords + 1).T)] = np.arange(n, dtype=np.int32)
    table = np.empty((n, 27), dtype=np.int32)
    for k, d in enumerate(OFFSETS):
        table[:, k] = grid[tuple((coords + 1 + d).T)]
    return table


def _child_offset_tables() -> tuple[np.ndarray, np.ndarray]:
    """For child slot ``s`` and kernel offset ``k``: the parent-neighbourhood offset
    ``pk`` holding that neighbour, and the neighbour's own slot ``s'``."""
    pk = np.zeros((8, 27), np.int32)
    ss = np.zeros((8, 27), np.int32)
    for s in range(8):
        bits = np.array([(s >> 2) & 1, (s >> 1) & 1, s & 1])
        for k, d in enumerate(OFFSETS):
            t = bits + d
            pd, cb = t >> 1, t & 1  # floor division and parity, valid for t in -1..2
            pk[s, k] = (pd[0] + 1) * 9 + (pd[1] + 1) * 3 + (pd[2] + 1)
            ss[s, k] = cb[0] * 4 + cb[1] * 2 + cb[2]
    return pk, ss


_CHILD_PK, _CHILD_SLOT = _child_offset_tables()


def child_neighbour_table(
    parent_table: mx.array, child_mask: np.ndarray
) -> tuple[mx.array, np.ndarray, np.ndarray]:
    """Neighbour table of the children produced by an octree upsampling.

    Child ``(p, s)`` of parent ``p`` sits at ``2 * p + bits(s)`` with
    ``bits(s) = (s >> 2 & 1, s >> 1 & 1, s & 1)``. Its neighbour at ``d`` has parent
    offset ``floor((bits + d) / 2)``, which always lies in the parent's own 3x3x3
    neighbourhood, so the child table follows from the parent table and the child
    masks without any hashing. The two gathers run on the GPU.

    Args:
        parent_table: ``[P, 27]`` parent neighbour table (``P`` for absent).
        child_mask: ``[P, 8]`` boolean occupancy of each child slot.

    Returns:
        ``(table [C, 27], parent [C], slot [C])`` in the decoder's child order
        (parent-major, slot-minor).
    """
    num_parents = child_mask.shape[0]
    ids = np.flatnonzero(child_mask.reshape(-1))
    num_children = int(ids.size)
    lookup = np.full(num_parents * 8 + 8, num_children, dtype=np.int32)
    lookup[ids] = np.arange(num_children, dtype=np.int32)
    parent, slot = ids // 8, ids % 8
    slot_mx = mx.array(slot.astype(np.int32))
    rows = mx.take(parent_table, mx.array(parent.astype(np.int32)), axis=0)  # [C, 27]
    neighbour_parent = mx.take_along_axis(rows, mx.take(mx.array(_CHILD_PK), slot_mx, axis=0), axis=1)
    index = neighbour_parent * 8 + mx.take(mx.array(_CHILD_SLOT), slot_mx, axis=0)
    table = mx.take(mx.array(lookup), index)
    mx.eval(table)
    return table, parent, slot


class Level:
    """One coordinate set and its convolution gather table (``N`` marks absent)."""

    def __init__(self, coords: np.ndarray, table: mx.array | np.ndarray):
        self.coords = coords
        self.n = int(coords.shape[0])
        self.table = table if isinstance(table, mx.array) else mx.array(table)
        self._table_np = table if isinstance(table, np.ndarray) else None

    @property
    def table_np(self) -> np.ndarray:
        if self._table_np is None:
            self._table_np = np.array(self.table)
        return self._table_np

    @classmethod
    def from_coords(cls, coords: np.ndarray) -> "Level":
        return cls(np.asarray(coords, dtype=np.int32), neighbour_table(coords))


ASSEMBLE_IN_PLACE_BYTES = 1 << 30
"""Blocked results larger than this are written into one preallocated array
(in-place slice updates) instead of concatenated, which would briefly hold the
result twice."""

DONATION_WAIT_S = 1e-3
"""Pause before each in-place update. MLX updates an array in place only if nothing
else holds its buffer, and the previous update's command buffer releases its hold
asynchronously after ``mx.eval`` returns. Without the pause, roughly half of the
1024³ decodes measured with the profiler thread running lost that race once and
copied a 2 GB activation (MLX peak 7.1 instead of 5.4 GB); with it, none did. Cost:
~0.3 s per 1024³ decode."""


def blocked(n: int, rows: int, block_fn, width: int, dtype: mx.Dtype, accumulate: list | None = None) -> mx.array:
    """``[n, width]`` result of ``block_fn(start, stop)`` over row blocks of ``rows``.

    Every block is evaluated before the next is built, so at most one block's
    temporaries are alive at a time.

    Args:
        accumulate: Optional one-element list holding the only reference to an
            ``[n, width]`` array ``a``; returns ``a + blocks``, added row block by row
            block into ``a``'s own buffer (``a.at[rows].add`` updates in place only
            when nothing else references ``a``, hence the list).
    """
    if accumulate is not None:
        out = accumulate.pop()
        for start in range(0, n, rows):
            stop = min(start + rows, n)
            block = block_fn(start, stop)
            mx.eval(block)
            time.sleep(DONATION_WAIT_S)
            out = out.at[start:stop].add(block)
            del block
            mx.eval(out)
        return out
    if rows >= n:
        return block_fn(0, n)
    if n * width * dtype.size > ASSEMBLE_IN_PLACE_BYTES:
        out = mx.zeros((n, width), dtype=dtype)
        for start in range(0, n, rows):
            stop = min(start + rows, n)
            block = block_fn(start, stop)
            mx.eval(block)
            time.sleep(DONATION_WAIT_S)
            out[start:stop] = block
            del block
            mx.eval(out)
        return out
    blocks = []
    for start in range(0, n, rows):
        blocks.append(block_fn(start, min(start + rows, n)))
        mx.eval(blocks[-1])
    return mx.concatenate(blocks, axis=0)


def pad_rows(x: mx.array) -> mx.array:
    """``x`` with one zero row appended: the target of absent neighbours (index ``N``)."""
    return mx.concatenate([x, mx.zeros((1, x.shape[1]), dtype=x.dtype)], axis=0)


def gather_neighbours(padded: mx.array, level: Level, start: int, stop: int) -> mx.array:
    """``[stop - start, 27, C]`` neighbour features of :func:`pad_rows` output."""
    return mx.take(padded, level.table[start:stop], axis=0)


def conv_weight(weight: mx.array) -> mx.array:
    """spconv ``(out, 3, 3, 3, in)`` -> GEMM layout ``[27 * in, out]``."""
    out_channels = weight.shape[0]
    return weight.reshape(out_channels, -1).T


def sparse_conv(
    x: mx.array, level: Level, weight_t: mx.array, bias: mx.array | None
) -> mx.array:
    """Submanifold 3x3x3 convolution as one gathered GEMM per row block.

    Args:
        x: ``[N, Cin]`` features; GEMM runs in ``weight_t.dtype``.
        level: Coordinate set of ``x`` (submanifold: output rows = input rows).
        weight_t: ``[27 * Cin, Cout]`` from :func:`conv_weight`.
        bias: ``[Cout]`` or ``None``.
    """
    padded = pad_rows(x.astype(weight_t.dtype))
    cin = x.shape[1]
    rows = max(1, GATHER_ELEMENTS // (27 * cin))
    if rows < level.n:
        # Evaluate block by block: left lazy, MLX would materialise every gathered
        # block before the GEMMs consume them (12 GB at the 512 level).
        mx.eval(padded)

    def block(start: int, stop: int) -> mx.array:
        y = gather_neighbours(padded, level, start, stop).reshape(stop - start, 27 * cin) @ weight_t
        return y if bias is None else y + bias

    return blocked(level.n, rows, block, weight_t.shape[1], weight_t.dtype)


def downsample_structure(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official ``SparseDownsample(2)`` index structure.

    Returns:
        ``(low_coords, inverse, counts)``: unique ``coords // 2`` in lexicographic
        ``(x, y, z)`` order (the order ``torch.unique`` gives the packed codes),
        each fine row's parent, and children per parent.
    """
    half = np.asarray(coords, dtype=np.int64) // 2
    extent = half.max(axis=0) + 1
    codes = (half[:, 0] * extent[1] + half[:, 1]) * extent[2] + half[:, 2]
    unique, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    low = np.stack(
        [unique // (extent[1] * extent[2]), (unique // extent[2]) % extent[1], unique % extent[2]],
        axis=1,
    ).astype(np.int32)
    return low, inverse.astype(np.int32), counts.astype(np.int32)
