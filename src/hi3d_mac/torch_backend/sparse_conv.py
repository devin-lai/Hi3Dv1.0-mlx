"""Pure-PyTorch submanifold sparse convolution exposing the spconv 2.3.6 API subset
that ``sparc3d.ops.sparse_backend`` uses.

Semantics follow spconv's own index generation (``spconv/csrc/sparse/indices.py``,
``ConvOutLocIter::nhw_to_npq`` and ``generate_subm_conv_inds``): for kernel offset
``r = (r0, r1, r2)`` with flat index ``k = r0*9 + r1*3 + r2`` an input voxel at
``p`` feeds the output voxel at ``p + center - r``. Equivalently

    out[o] = bias + sum_r W[:, r0, r1, r2, :] @ in[o + r - center],

a cross-correlation over the ``[batch, x, y, z]`` coordinate columns with the
``(out, kx, ky, kz, in)`` weight layout. ``tests/test_backends.py`` checks this
against a dense ``torch.nn.functional.conv3d``.
"""

from __future__ import annotations

import enum
import itertools
from typing import Any, Sequence

import torch
from torch import nn

GATHER_BYTES = 256 * 1024 * 1024
"""Upper bound on one gathered feature block, keeping the 1024-level decode bounded."""


class ConvAlgo(enum.Enum):
    """spconv algorithm selector; every value maps to the same gather/GEMM path."""

    Native = 0
    MaskImplicitGemm = 1
    MaskSplitImplicitGemm = 2


class _PairTable:
    """``[2, K, N]``-indexable view over per-offset pair lists.

    spconv's Native layout pads every offset to ``N`` rows. Code only ever reads
    ``table[side, offset, :count]``, so the lists are kept unpadded instead, which
    avoids a ``2 * 27 * N`` int32 allocation at the 1024 level.
    """

    def __init__(self, sources: list[torch.Tensor], targets: list[torch.Tensor]):
        self._sides = (sources, targets)

    def __getitem__(self, index: tuple[int, int, slice]) -> torch.Tensor:
        side, offset, rows = index
        return self._sides[side][offset][rows]


class IndiceData:
    """Cached submanifold pairs, mirroring ``spconv.pytorch.core.IndiceData``."""

    def __init__(
        self,
        out_indices: torch.Tensor,
        indices: torch.Tensor,
        indice_pairs: Any,
        indice_pair_num: torch.Tensor,
        spatial_shape: Sequence[int],
        out_spatial_shape: Sequence[int],
        is_subm: bool = True,
        algo: ConvAlgo = ConvAlgo.Native,
        **_: Any,
    ):
        self.out_indices = out_indices
        self.indices = indices
        self.indice_pairs = indice_pairs
        self.indice_pair_num = indice_pair_num
        self.spatial_shape = list(spatial_shape)
        self.out_spatial_shape = list(out_spatial_shape)
        self.is_subm = is_subm
        self.algo = algo


class SparseConvTensor:
    """Features, ``[N, 4]`` int32 ``(batch, x, y, z)`` indices and a shared pair cache."""

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: Sequence[int],
        batch_size: int,
        indice_dict: dict[str, IndiceData] | None = None,
    ):
        self._features = features
        self.indices = indices
        self.spatial_shape = [int(v) for v in spatial_shape]
        self.batch_size = int(batch_size)
        self.indice_dict = {} if indice_dict is None else indice_dict

    @property
    def features(self) -> torch.Tensor:
        return self._features

    def replace_feature(self, feature: torch.Tensor) -> "SparseConvTensor":
        """Return a tensor over the same coordinates that shares the pair cache."""
        return SparseConvTensor(
            feature, self.indices, self.spatial_shape, self.batch_size, self.indice_dict
        )

    def find_indice_pair(self, key: str | None) -> IndiceData | None:
        return None if key is None else self.indice_dict.get(key)


def _voxel_keys(indices: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    batch, x, y, z = indices.long().unbind(1)
    sx, sy, sz = (int(v) for v in spatial_shape)
    return ((batch * sx + x) * sy + y) * sz + z


def subm_pairs(
    indices: torch.Tensor, spatial_shape: Sequence[int], kernel_size: int = 3
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Per-offset ``(input row, output row)`` pairs of a submanifold convolution.

    Returns:
        ``(sources, targets, counts)`` with one int64 tensor per flat kernel offset.
    """
    center = kernel_size // 2
    coords = indices.long()
    shape = [int(v) for v in spatial_shape]
    keys = _voxel_keys(coords, shape)
    sorted_keys, order = torch.sort(keys)
    arange = torch.arange(coords.shape[0], device=coords.device)
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for r in itertools.product(range(kernel_size), repeat=3):
        if r == (center, center, center):
            sources.append(arange)
            targets.append(arange)
            continue
        shift = torch.tensor(
            [0, center - r[0], center - r[1], center - r[2]], device=coords.device
        )
        neighbour = coords + shift
        inside = torch.ones_like(neighbour[:, 0], dtype=torch.bool)
        for axis in range(3):
            inside &= (neighbour[:, axis + 1] >= 0) & (neighbour[:, axis + 1] < shape[axis])
        query = _voxel_keys(neighbour, shape)
        position = torch.searchsorted(sorted_keys, query).clamp_(max=keys.shape[0] - 1)
        found = inside & (sorted_keys[position] == query)
        sources.append(arange[found])
        targets.append(order[position[found]])
    counts = torch.tensor([len(s) for s in sources], dtype=torch.int32)
    return sources, targets, counts


def _accumulate(
    output: torch.Tensor,
    features: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    rows = max(1, GATHER_BYTES // max(1, features.shape[1] * features.element_size()))
    for start in range(0, source.shape[0], rows):
        stop = start + rows
        output.index_add_(0, target[start:stop], features[source[start:stop]] @ weight.T)


def subm_conv(
    features: torch.Tensor,
    pairs: IndiceData,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a submanifold convolution given cached pairs.

    Accumulation order matches the official chunked fallback
    (``SubMConv3d._large_forward``): the centre tap, then offset ``k`` and its mirror
    ``K - 1 - k`` for ``k = 0 .. K/2 - 1``.
    """
    out_channels = weight.shape[0]
    kernel_volume = weight[0, ..., 0].numel()
    weights = weight.reshape(out_channels, kernel_volume, weight.shape[-1])
    center = kernel_volume // 2
    output = features @ weights[:, center, :].T
    for offset in range(center):
        for k in (offset, kernel_volume - 1 - offset):
            count = int(pairs.indice_pair_num[k])
            if count:
                _accumulate(
                    output,
                    features,
                    pairs.indice_pairs[0, k, :count],
                    pairs.indice_pairs[1, k, :count],
                    weights[:, k, :],
                )
    if bias is not None:
        output = output + bias
    return output


class SubMConv3d(nn.Module):
    """Submanifold 3D convolution with spconv's parameter names and layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        indice_key: str | None = None,
        algo: ConvAlgo | None = None,
        **_: Any,
    ):
        super().__init__()
        ks = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        if len(set(ks)) != 1 or ks[0] % 2 == 0:
            raise ValueError(f"only cubic odd kernels are supported, got {ks}")
        if groups != 1:
            raise ValueError("grouped sparse convolution is not supported")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = ks
        self.stride = (1, 1, 1)
        self.padding = tuple(k // 2 for k in ks)
        self.dilation = (1, 1, 1)
        self.output_padding = (0, 0, 0)
        self.indice_key = indice_key
        self.algo = algo or ConvAlgo.MaskImplicitGemm
        self.weight = nn.Parameter(torch.empty(out_channels, *ks, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _pairs(self, input: SparseConvTensor) -> IndiceData:
        data = input.find_indice_pair(self.indice_key)
        if data is None:
            data = get_indice_pair_data(input, self.kernel_size[0])
            if self.indice_key is not None:
                input.indice_dict[self.indice_key] = data
        return data

    def forward(self, input: SparseConvTensor) -> SparseConvTensor:
        data = self._pairs(input)
        output = subm_conv(input.features, data, self.weight, self.bias)
        return input.replace_feature(output)


def get_indice_pair_data(input: SparseConvTensor, kernel_size: int = 3) -> IndiceData:
    sources, targets, counts = subm_pairs(input.indices, input.spatial_shape, kernel_size)
    return IndiceData(
        input.indices,
        input.indices,
        _PairTable(sources, targets),
        counts,
        input.spatial_shape,
        input.spatial_shape,
        is_subm=True,
        algo=ConvAlgo.Native,
    )


def get_indice_pairs(
    indices: torch.Tensor,
    batch_size: int,
    spatial_shape: Sequence[int],
    algo: ConvAlgo,
    ksize: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    out_padding: Sequence[int],
    subm: bool = False,
    transpose: bool = False,
) -> tuple[torch.Tensor, _PairTable, torch.Tensor]:
    """``spconv.pytorch.ops.get_indice_pairs`` for the submanifold case only."""
    if not subm or transpose:
        raise NotImplementedError("only submanifold index pairs are implemented")
    if tuple(stride) != (1, 1, 1) or tuple(dilation) != (1, 1, 1):
        raise NotImplementedError("only stride 1 / dilation 1 are implemented")
    sources, targets, counts = subm_pairs(indices, spatial_shape, int(ksize[0]))
    return indices, _PairTable(sources, targets), counts
