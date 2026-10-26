"""MLX port of the SparconvVAE decoder (``PTv3MeshDecoder``).

Follows ``mac/upstream/sparc3d/models/autoencoders/{ptv3,blocks}.py`` and
``structures/point.py``: a serialized-attention stack at the latent grid, then
three residual stages that double the resolution and keep the eight children each
voxel's occupancy head predicts.

Index structures per level come from :mod:`.sparse`; children inherit their
neighbour table from the parent table on the GPU (no hashing of millions of
voxels). Every ``conv(swish(GroupNorm(x)))`` gathers, normalises and activates
its input block in one Metal kernel, so the activated tensor and its padded copy
are never stored; large outputs are assembled in place. GroupNorm statistics,
child projections and the per-level index tables all stay on the GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from . import kernels, sparse
from .sparse import Level

F32 = mx.float32

SKIP_POINTS = 12_000_000
"""Upstream ``_RESNET_SKIP_POINTS``: residual blocks above this size are no-ops."""

PATCH = 48
ORDERS = ("z", "z-trans")

STATS_ROWS = 1 << 19
"""Rows per block of the GroupNorm moment sums."""

CHILD_ELEMENTS = 1 << 26
"""Elements of one block of child projections ``[parents, 8C]``."""

EVAL_POINTS = 500_000
"""Above this many points every layer is evaluated eagerly, so the lazy graph never
holds several full-resolution activations at once."""


# ----------------------------------------------------------------------------- helpers


def morton(coords: np.ndarray, depth: int) -> np.ndarray:
    """``xyz2key``: interleave bits, x highest (bit 3i+2), y (3i+1), z (3i)."""
    x, y, z = (coords[:, i].astype(np.int64) for i in range(3))
    key = np.zeros_like(x)
    for i in range(depth):
        key |= ((x >> i) & 1) << (3 * i + 2) | ((y >> i) & 1) << (3 * i + 1) | ((z >> i) & 1) << (3 * i)
    return key


def serialization(coords: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """``Point.serialization(order=("z", "z-trans"))`` for one batch element."""
    depth = int(coords.max()).bit_length()
    out = []
    for order in ORDERS:
        c = coords if order == "z" else coords[:, [1, 0, 2]]
        code = morton(c, depth)  # batch 0 adds nothing above the coordinate bits
        perm = np.argsort(code, kind="stable")
        inverse = np.empty_like(perm)
        inverse[perm] = np.arange(perm.shape[0])
        out.append((perm, inverse))
    return out


def window_index(perm: np.ndarray, inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``SerializedAttention.get_padding_and_inverse`` for one batch element.

    Returns:
        ``(gather [W * L], scatter [N])``: rows in padded window order, and each
        point's position in that order. The last partial window is completed with
        the preceding rows, so it covers the last ``PATCH`` serialized points.
    """
    n = perm.shape[0]
    if n <= PATCH:
        return perm, inverse
    n_pad = -(-n // PATCH) * PATCH
    pad = np.arange(n_pad)
    rem = n % PATCH
    if rem:
        pad[n_pad - PATCH + rem :] = pad[n_pad - 2 * PATCH + rem : n_pad - PATCH]
    return perm[pad], inverse


@dataclass
class DecodedPoints:
    """Decoder output in the layout ``point_to_mesh`` consumes."""

    grid_coord: np.ndarray  # [N, 3] int32 at ``resolution``
    feat: np.ndarray  # [N, 32] activated: tanh(sdf) 8 | tanh(deform) 24
    sdf_sign: np.ndarray  # [N, 8] logits
    resolution: int
    matrix: np.ndarray  # [4, 4] grid-to-coord transform of the latent grid


class SparconvVAEDecoderMLX:
    def __init__(self, artifact, dtype: mx.Dtype = F32, tensors=None):
        """
        Args:
            dtype: GEMM dtype of the convolutions and linears. ``float32`` is the
                exact decoder. ``float16`` is the official CUDA autocast precision:
                fp16 GEMMs with fp32 normalisation statistics, and the activations
                between layers stored in fp16, which halves the decoder's memory.
        """
        if tensors is None:
            tensors = mx.load(str(artifact / "model.safetensors"))
        self.dt = dtype
        self.act = F32 if dtype == F32 else mx.float16
        """Storage dtype of the activations between decoder layers."""
        self.p: dict[str, mx.array] = {}
        for name, w in tensors.items():
            name = name.removeprefix("decoder.")
            if name.endswith("inv_grid_coord"):
                continue
            if w.ndim == 5:  # sparse conv (out, 3, 3, 3, in) -> [27*in, out]
                w = sparse.conv_weight(w)
            # GEMM weights (sparse convs and linears) use ``dtype``; group/layer
            # norms and biases stay fp32. The octree occupancy heads decide which
            # voxels survive, so they keep fp32 weights too.
            gemm = name.endswith(".weight") and w.ndim == 2 and "cls_head" not in name
            self.p[name] = w.astype(dtype if gemm else F32)
        mx.eval(self.p)

    # -- layers ---------------------------------------------------------------------

    def linear(self, x: mx.array, name: str, bias: bool = True) -> mx.array:
        w = self.p[f"{name}.weight"]
        y = x.astype(w.dtype) @ w.T
        return y + self.p[f"{name}.bias"] if bias else y

    def conv(self, x: mx.array, level: Level, name: str) -> mx.array:
        w = self.p[f"{name}.weight"]
        return sparse.sparse_conv(x, level, w, None).astype(F32) + self.p[f"{name}.bias"]

    @staticmethod
    def group_stats(x: mx.array, eps: float = 1e-6) -> tuple[mx.array, mx.array]:
        """``PointGroupNorm`` (one batch element) mean and ``rsqrt(var + eps)`` per
        channel, in fp32.

        Per-channel column sums first, then per group: the same moments as
        upstream's per-group reduction (only the fp32 summation order differs),
        without a strided reduction over ``[N, G, C/G]``.
        """
        n, c = x.shape
        groups = 32 if 4 * 32 <= c else c // 4
        width = c // groups
        # Row blocks bound the fp32 temporaries (x in fp32, x * x) to one block.
        s1 = s2 = mx.zeros((c,), dtype=F32)
        for start in range(0, n, STATS_ROWS):
            xf = x[start : start + STATS_ROWS].astype(F32)
            s1, s2 = s1 + mx.sum(xf, axis=0), s2 + mx.sum(xf * xf, axis=0)
            if n > STATS_ROWS:
                mx.eval(s1, s2)
        count = float(n * width)
        mean = s1.reshape(groups, width).sum(axis=1) / (count + eps)
        var = s2.reshape(groups, width).sum(axis=1) / (count + eps) - mean * mean
        inv = mx.rsqrt(var + eps)
        return mx.repeat(mean, width), mx.repeat(inv, width)

    def conv_gn_swish(self, x: mx.array, level: Level, norm: str, name: str, out: mx.Dtype | None = None,
                      residual: list | None = None) -> mx.array:
        """``conv(swish(GroupNorm(x)))`` with the activation applied while gathering.

        The normalised, activated input is never stored: each block of neighbour
        rows is normalised and activated by one Metal kernel straight into the
        block the GEMM reads. Same arithmetic as ``conv(swish(group_norm(x)))``.

        With ``residual`` (a one-element list holding the only reference to an array
        ``r``), returns ``r + conv(...)``, added block by block into ``r``'s buffer
        (the convolution only gathers from ``x``).
        """
        w = self.p[f"{name}.weight"]  # [27 * Cin, Cout]
        bias = self.p[f"{name}.bias"]
        mean, inv = self.group_stats(x)
        mx.eval(x, mean, inv)
        rows = max(1, sparse.GATHER_ELEMENTS // (27 * x.shape[1]))

        if out is None:  # the occupancy heads keep fp32 weights and fp32 logits
            out = F32 if w.dtype == F32 else self.act

        def block(start: int, stop: int) -> mx.array:
            g = kernels.gather_group_norm_swish(
                x, level.table[start:stop], mean, inv, self.p[f"{norm}.weights"], self.p[f"{norm}.bias"], w.dtype)
            return ((g @ w).astype(F32) + bias).astype(out)

        return sparse.blocked(level.n, rows, block, w.shape[1], out, accumulate=residual)

    @staticmethod
    def _barrier(x: mx.array) -> mx.array:
        if x.shape[0] > EVAL_POINTS:
            mx.eval(x)
        return x

    def resnet(self, holder: list, level: Level, name: str) -> mx.array:
        """Residual block on ``holder.pop()``.

        The input is passed in a list the caller no longer references, so at large
        levels an identity-shortcut block can add its second convolution into the
        input's buffer: two full-size activations alive instead of three (2 GB less
        at 1024³ with the FP32 decoder). Same arithmetic as ``x + conv(...)``.
        """
        x = holder.pop()
        if x.shape[0] > SKIP_POINTS:
            return x
        h = self._barrier(self.conv_gn_swish(x, level, f"{name}.block.0", f"{name}.block.2"))
        if f"{name}.res_conv.weight" not in self.p and x.shape[0] > EVAL_POINTS and x.dtype == self.act:
            holder.append(x)
            del x
            return self.conv_gn_swish(h, level, f"{name}.block.3", f"{name}.block.5", residual=holder)
        h = self._barrier(self.conv_gn_swish(h, level, f"{name}.block.3", f"{name}.block.5"))
        shortcut = self.linear(x, f"{name}.res_conv").astype(self.act) if f"{name}.res_conv.weight" in self.p else x
        return self._barrier(shortcut + h)

    def children(self, x: mx.array, parent: np.ndarray, slot: np.ndarray, name: str) -> mx.array:
        """``SerializedUnpooling`` features of the kept children only.

        Upstream projects every voxel to all eight children (``Linear(C, 8C)``) and
        then drops the unoccupied ones; child ``(p, s)`` is row ``p * 8 + s`` of that
        projection viewed as ``[8P, C]``. Children are parent-major, so a block of
        parents owns a contiguous block of children: project the parents block by
        block and gather their kept children, never holding the whole ``[P, 8C]``.
        """
        w = self.p[f"{name}.weight"]  # [8C, C]
        c = x.shape[1]
        num_parents = x.shape[0]
        rows = max(1, CHILD_ELEMENTS // (8 * c))
        flat = parent.astype(np.int64) * 8 + slot
        bounds = np.searchsorted(parent, np.arange(0, num_parents + rows, rows))
        pieces = []
        for i, p0 in enumerate(range(0, num_parents, rows)):
            c0, c1 = bounds[i], bounds[i + 1]
            if c1 == c0:
                continue
            proj = (x[p0 : p0 + rows].astype(w.dtype) @ w.T).reshape(-1, c)
            pieces.append(mx.take(proj, mx.array((flat[c0:c1] - p0 * 8).astype(np.int32)), axis=0).astype(self.act))
            if rows < num_parents:
                mx.eval(pieces[-1])
        return self._barrier(pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=0))

    def layer_norm(self, x: mx.array, name: str) -> mx.array:
        return mx.fast.layer_norm(x.astype(F32), self.p[f"{name}.weight"], self.p[f"{name}.bias"], 1e-5)

    def attention_block(self, x, level, name, windows):
        gather, scatter = windows
        h = self.conv(x, level, f"{name}.cpe.0")
        h = self.linear(h, f"{name}.cpe.1").astype(F32)
        x = x + self.layer_norm(h, f"{name}.cpe.2")
        h = self.layer_norm(x, f"{name}.norm1.0")
        qkv = self.linear(h, f"{name}.attn.qkv")
        heads, c = 32, x.shape[1]
        g = mx.take(qkv, gather, axis=0).reshape(-1, PATCH if gather.shape[0] >= PATCH else gather.shape[0], 3, heads, c // heads)
        q, k, v = (g[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=(c // heads) ** -0.5)
        o = o.transpose(0, 2, 1, 3).reshape(-1, c)
        o = mx.take(o, scatter, axis=0)
        x = x + self.linear(o, f"{name}.attn.proj").astype(F32)
        h = self.layer_norm(x, f"{name}.norm2.0")
        h = self.linear(h, f"{name}.mlp.0.fc1").astype(F32)
        h = 0.5 * h * (1.0 + mx.erf(h / math.sqrt(2.0)))
        return x + self.linear(h, f"{name}.mlp.0.fc2").astype(F32)

    def occupancy(self, x: mx.array, level: Level, name: str) -> mx.array:
        return self.conv_gn_swish(x, level, f"{name}.cls_head.fc.0", f"{name}.cls_head.fc.2")

    # -- forward --------------------------------------------------------------------

    def __call__(
        self,
        coords: np.ndarray,
        latents: np.ndarray,
        resolution: int,
        force_masks: list[np.ndarray] | None = None,
    ) -> DecodedPoints:
        """Decode latents on ``coords`` (``[N, 3]`` at ``resolution``).

        Args:
            force_masks: Optional per-upsampling ``[P, 8]`` child masks to use instead
                of this decoder's own occupancy decisions (validation only).
        """
        self.masks: list[np.ndarray] = []
        coords = np.asarray(coords, dtype=np.int32)
        grid = coords.astype(np.float32)
        center = grid.mean(axis=0)
        scale = np.linalg.norm(grid - center, axis=1).max() + 1e-8
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] /= scale
        matrix[:3, 3] = -center / scale

        level = Level.from_coords(coords)
        x = self.conv(mx.array(latents.astype(np.float32)), level, "z_to_f.conv.0")
        serial = serialization(coords)
        x = self.resnet([x], level, "mid.block_in")
        for i in range(2):
            gather, scatter = window_index(*serial[i % len(ORDERS)])
            x = self.attention_block(x, level, f"mid.block{i}", (mx.array(gather.astype(np.int32)), mx.array(scatter.astype(np.int32))))
        x = self.resnet([x], level, "mid.block_out")
        mx.eval(x)

        res = resolution
        for stage in (3, 2, 1, 0):
            name = f"dec.dec{stage}"
            holder = [x]
            del x
            x = self.resnet(holder, level, f"{name}.block0")
            holder = [x]
            del x
            x = self.resnet(holder, level, f"{name}.block1")
            if stage == 0:
                break
            logits = self.occupancy(x, level, f"{name}.up.upsample")
            mx.eval(logits)
            mask = np.array(logits > 0)
            if force_masks is not None:
                mask = force_masks[len(self.masks)]
            self.masks.append(mask)
            table, parent, slot = sparse.child_neighbour_table(level.table, mask)
            x = self.children(x, parent, slot, f"{name}.up.upsample.proj")
            bits = np.stack([(slot >> 2) & 1, (slot >> 1) & 1, slot & 1], 1)
            coords = (level.coords[parent] * 2 + bits).astype(np.int32)
            level = Level(coords, table)
            res *= 2
            mx.eval(x)

        out = self.conv_gn_swish(x, level, "conv_out.conv.0", "conv_out.conv.2", out=F32)
        out = mx.concatenate([out[:, :8], mx.tanh(out[:, 8:])], axis=1)
        out = np.array(out)
        return DecodedPoints(level.coords, out[:, 8:], out[:, :8], res, matrix)
