"""MLX port of ``sparc3d.models.transformers.Sparc3DTransformer``.

Mirrors the upstream forward pass operation by operation; see
``mac/upstream/sparc3d/models/transformers/sparc3d_transformer.py`` and
``sparc3d/nn``. Precision is parameterised by :class:`Precision`:

* ``dtype``: activation/GEMM dtype of the torso. ``float16`` reproduces the
  released mixed precision; ``float32`` (with ``MLX_ENABLE_TF32=0``) is exact
  FP32.
* LayerNorm / RMSNorm / RoPE / softmax statistics always run in float32, as
  upstream does.

Exact restructurings (identical mathematics, fewer FLOPs):

* Cross-attention keys/values depend only on the image condition; they are
  projected once per pass instead of once per evaluation.
* With the all-zero negative condition every key and value equals the
  ``to_kv`` bias, so attention returns the value bias exactly; the negative
  branch's cross-attention is the constant ``to_out(v_bias)``.
* The input stage does not see the condition, so the positive and negative
  classifier-free-guidance branches share it.
* ``out_blocks.0`` upsamples by gathering parent rows; its row-wise norm, SiLU and
  skip projection are applied before the gather, and its first convolution is
  evaluated as per-offset parent projections (see :func:`_up_conv`).

Fused kernels (same arithmetic, fewer passes over memory; :mod:`.kernels`):

* RoPE + per-head QK RMSNorm + the ``[B, H, N, D]`` head layout in one Metal
  kernel straight from the fused QKV projection (was ~18% of a block).
* LayerNorm + adaLN modulation in one kernel for fp16 rows, with the same two
  fp16 roundings as the composed ops.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import attention, kernels, sparse
from .sparse import Level

F32 = mx.float32


@dataclass(frozen=True)
class Precision:
    dtype: mx.Dtype = mx.float16
    """GEMM / attention dtype of the torso (weights are stored in it)."""
    residual: mx.Dtype | None = None
    """Residual-stream dtype; ``None`` keeps it in ``dtype`` as upstream does."""
    attention: str = "flash"
    """fp16 attention kernel: ``flash`` (:mod:`.attention`, the neural-accelerator
    kernel) or ``mlx`` (``mx.fast.scaled_dot_product_attention``). float32 always
    uses MLX's kernel."""
    gemm: str = "mlx"
    """``w8a8`` opts into native int8 activations/weights with signed H64 rotation."""
    optimized: bool = False
    """Coalesced QKV preprocessing and shared first-block self-attention for CFG.
    Keeps FP32 statistics, FP16 GEMMs and the original sampling schedule."""
    preserve_fp16: bool = False
    """Retain the original reduction order and GEMM shapes. Optimized scheduling
    may share first-block attention, but not change projection batch sizes."""


# ----------------------------------------------------------------------------- weights


def gemm_layout(name: str, w: mx.array) -> mx.array:
    """Sparse conv ``(out, 3, 3, 3, in)`` -> ``[out, 27*in]``; other tensors unchanged."""
    return sparse.conv_weight(w).T if name.endswith("conv.weight") and w.ndim == 5 else w


class Weights:
    """Inference parameter store for released FP16 and pre-quantized weights.

    Accepts the official checkpoint (fp16 torso, fp32 rest), or a released file that holds
    ``<name>.weight`` (packed), ``<name>.scales`` and ``<name>.biases``.
    """

    def __init__(self, tensors: dict[str, mx.array], precision: Precision, quant_meta: dict | None = None):
        self.p: dict[str, mx.array] = {}
        self.q: dict[str, tuple] = {}
        self.precision = precision
        dt = precision.dtype
        quant_meta = quant_meta or {}
        for name, w in tensors.items():
            if name.endswith((".scales", ".biases")):
                continue
            base = name[: -len(".weight")] if name.endswith(".weight") else None
            if base is not None and f"{base}.scales" in tensors:
                bits, group = quant_meta[name]
                self.q[name] = (w, tensors[f"{base}.scales"], tensors[f"{base}.biases"], group, bits)
                continue
            w = gemm_layout(name, w)
            if w.dtype == mx.float16:
                w = w.astype(dt)
            self.p[name] = w
        mx.eval(self.p, self.q)

    def _qmm(self, x: mx.array, name: str) -> mx.array:
        wq, scales, biases, group, bits = self.q[name]
        return mx.quantized_matmul(
            x.astype(scales.dtype), wq, scales, biases, transpose=True, group_size=group, bits=bits
        )

    def linear(self, x: mx.array, name: str, bias: bool = True) -> mx.array:
        """``x @ W.T + b`` in the weight's dtype (fp32 accumulation for fp16)."""
        b = self.p.get(f"{name}.bias") if bias else None
        if f"{name}.weight" in self.q:
            y = self._qmm(x, f"{name}.weight")
            return y if b is None else y + b.astype(y.dtype)
        w = self.p[f"{name}.weight"]
        x = x.astype(w.dtype)
        return mx.addmm(b.astype(w.dtype), x, w.T) if b is not None else x @ w.T

    def conv(self, x: mx.array, level: Level, name: str) -> mx.array:
        """Submanifold convolution whose weight was reshaped to ``[Cout, 27*Cin]``."""
        key = f"{name}.weight"
        if key in self.q:
            cin = x.shape[1]
            rows = max(1, sparse.GATHER_ELEMENTS // (27 * cin))
            padded = sparse.pad_rows(x.astype(self.q[key][1].dtype))
            outs = []
            chunked = rows < level.n
            for s in range(0, level.n, rows):
                e = min(s + rows, level.n)
                g = sparse.gather_neighbours(padded, level, s, e).reshape(e - s, 27 * cin)
                outs.append(self._qmm(g, key))
                if chunked:
                    mx.eval(outs[-1])
            y = outs[0] if len(outs) == 1 else mx.concatenate(outs, 0)
            return y + self.p[f"{name}.bias"].astype(y.dtype)
        w = self.p[key]
        return sparse.sparse_conv(x, level, w.T, self.p[f"{name}.bias"].astype(w.dtype))

    def dense(self, name: str) -> mx.array:
        """Dequantized (or plain) weight, for one-off restructured products."""
        if name in self.q:
            wq, scales, biases, group, bits = self.q[name]
            return mx.dequantize(wq, scales, biases, group, bits)
        return self.p[name]

    def nbytes(self) -> int:
        total = sum(v.nbytes for v in self.p.values())
        return total + sum(sum(t.nbytes for t in q[:3]) for q in self.q.values())


def load_config(path: Path) -> dict:
    return json.loads((path / "config.json").read_text())


# ----------------------------------------------------------------------------- math


def layer_norm32(x: mx.array, weight=None, bias=None, eps: float = 1e-6) -> mx.array:
    """``LayerNorm32``: statistics and affine in fp32, output in ``x.dtype``.

    MLX's kernel already accumulates fp16 input in fp32 (and applies fp32 affine
    parameters in fp32), so fp16 input skips the cast round trip; the output is
    bitwise identical.
    """
    if x.dtype == mx.float16:
        return mx.fast.layer_norm(x, weight, bias, eps).astype(x.dtype)
    return mx.fast.layer_norm(x.astype(F32), weight, bias, eps).astype(x.dtype)


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def gelu_tanh(x: mx.array) -> mx.array:
    xf = x.astype(F32)
    y = 0.5 * xf * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (xf + 0.044715 * xf * xf * xf)))
    return y.astype(x.dtype)


def timestep_embedding(t: mx.array, dim: int = 256, max_period: int = 10000) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=F32) / half)
    args = t[:, None].astype(F32) * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def rope_tables(coords: np.ndarray, head_dim: int = 96) -> tuple[mx.array, mx.array]:
    """cos/sin ``[N, head_dim/2]`` of ``RotaryPositionEmbedder`` for ``coords``."""
    freq_dim = head_dim // 3 // 2
    freqs = 1.0 / (10000.0 ** (np.arange(freq_dim, dtype=np.float32) / np.float32(freq_dim)))
    freqs = freqs.astype(np.float32)
    max_length = int(coords.max())
    if max_length >= 64:
        train, infer = 64, (64 + max_length) // 2
        wavelengths = freqs * train / (2 * math.pi)
        ratio = (wavelengths - 1) / (24 - 1)
        ratio = ratio + (1 - ratio) * train / infer
        ratio = np.where(wavelengths > 24, 1.0, ratio)
        ratio = np.where(wavelengths < 1, train / infer, ratio)
        freqs = (freqs * ratio).astype(np.float32)
    phases = coords.astype(np.float32).reshape(-1)[:, None] * freqs[None, :]
    phases = phases.reshape(coords.shape[0], 3 * freq_dim)
    pad = head_dim // 2 - phases.shape[1]
    cos, sin = np.cos(phases), np.sin(phases)
    if pad > 0:
        cos = np.concatenate([cos, np.ones((cos.shape[0], pad), np.float32)], 1)
        sin = np.concatenate([sin, np.zeros((sin.shape[0], pad), np.float32)], 1)
    return mx.array(cos.astype(np.float32)), mx.array(sin.astype(np.float32))


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate adjacent channel pairs of ``[N, H, D]`` (complex multiply in fp32)."""
    n, h, d = x.shape
    xf = x.astype(F32).reshape(n, h, d // 2, 2)
    re, im = xf[..., 0], xf[..., 1]
    c, s = cos[:, None, :], sin[:, None, :]
    out = mx.stack([re * c - im * s, re * s + im * c], axis=-1)
    return out.reshape(n, h, d)


def rms_norm_heads(x: mx.array, gamma: mx.array, out_dtype) -> mx.array:
    """``F.normalize(x) * gamma * sqrt(D)`` over the head dimension, in fp32."""
    xf = x.astype(F32)
    norm = mx.sqrt(mx.sum(xf * xf, axis=-1, keepdims=True))
    y = xf / mx.maximum(norm, 1e-12) * gamma * math.sqrt(x.shape[-1])
    return y.astype(out_dtype)


# ----------------------------------------------------------------------------- geometry


@dataclass
class Geometry:
    """Index structures of one sampling pass (coordinates never change within it)."""

    coords: np.ndarray
    full: Level = field(init=False)
    low: Level = field(init=False)

    def __post_init__(self):
        coords = np.asarray(self.coords, dtype=np.int32)
        self.full = Level.from_coords(coords)
        low, inverse, counts = sparse.downsample_structure(coords)
        self.low = Level.from_coords(low)
        self.inverse_np = inverse
        self.inverse = mx.array(inverse)
        # Upstream's scatter_reduce(mean) includes the zero-initialised output, so
        # each mean divides by (children + 1).
        self.mean_divisor = mx.array((counts + 1).astype(np.float32))[:, None]
        self.rope_cos, self.rope_sin = rope_tables(low)
        # Gather table for the restructured out_blocks.0 conv: the parent row of
        # each fine neighbour, flattened with its offset (N_low * 27 = absent).
        parent_of = np.append(inverse, self.low.n).astype(np.int64)  # absent -> N_low
        nb_parent = parent_of[self.full.table_np]
        flat = nb_parent * 27 + np.arange(27)[None, :]
        flat[nb_parent == self.low.n] = self.low.n * 27
        self.up_conv_index = mx.array(flat.astype(np.int32))

    def downsample(self, x: mx.array) -> mx.array:
        sums = mx.zeros((self.low.n, x.shape[1]), dtype=F32).at[self.inverse].add(x.astype(F32))
        return (sums / self.mean_divisor).astype(x.dtype)

    def upsample(self, x: mx.array) -> mx.array:
        return mx.take(x, self.inverse, axis=0)


# ----------------------------------------------------------------------------- model


class Sparc3DTransformerMLX:
    """Inference-only sparse flow transformer."""

    def __init__(self, artifact: Path, precision: Precision = Precision(), weights_file: Path | None = None):
        """
        Args:
            artifact: Official ``transformer_*`` directory (config and fp16 checkpoint).
            precision: Torso compute dtype.
            weights_file: Optional released pre-quantized checkpoint.
        """
        self.config = load_config(artifact)
        self.weights_path = weights_file or artifact / "model.safetensors"
        self._released_names: set[str] = set()
        self._schedule_cache: dict[bytes, tuple] = {}
        self._prepared_condition = None
        self._prepared_kv = None
        self.preparation_stats: dict = {}
        quant_meta = None
        if weights_file is None:
            tensors = mx.load(str(artifact / "model.safetensors"))
        else:
            tensors, metadata = mx.load(str(weights_file), return_metadata=True)
            if metadata.get("format") == "hi3d-w8a8-h64":
                if precision.gemm != "w8a8" or precision.dtype != mx.float16:
                    raise ValueError("W8A8 weights require the w8a8 GEMM mode and fp16 torso")
            else:
                quant_meta = {k: tuple(v) for k, v in json.loads(metadata["layout"]).items()}
        if precision.gemm == "w8a8" and "blocks.0.self_attn.to_qkv.w8_scale" not in tensors:
            raise ValueError("W8A8 requires the released Hi3Dv1.0-mlx-w8a8 checkpoints")
        self.w = Weights(tensors, precision, quant_meta)
        del tensors
        self.precision = precision
        self.dt = precision.dtype
        self.rt = precision.residual or precision.dtype
        self.num_blocks = self.config["num_blocks"]
        self.heads = self.config["num_heads"]
        self.channels = self.config["model_channels"]
        self.head_dim = self.channels // self.heads
        self.latent_mean = mx.array(np.array(self.config["latent_mean"], np.float32))
        self.latent_std = mx.array(np.array(self.config["latent_std"], np.float32))
        self._negative_cross = None
        self.compile = True
        self._block_fns: dict[tuple, object] = {}
        self.block_weights = [self._collect_block(i) for i in range(self.num_blocks)]
        self._block_qmeta = {
            key[len("blocks.0.") :]: q[3:] for key, q in self.w.q.items() if key.startswith("blocks.0.")
        }

    def _collect_block(self, i: int) -> dict:
        """Block ``i``'s parameters as a flat dict (arrays or quantized triples)."""
        prefix = f"blocks.{i}."
        out = {}
        for key, value in self.w.p.items():
            if key.startswith(prefix) and ".cross_attn.to_kv." not in key:
                out[key[len(prefix) :]] = value
        for key, q in self.w.q.items():
            if key.startswith(prefix) and ".cross_attn.to_kv." not in key:
                out[key[len(prefix) :]] = q[:3]
        return out

    # -- per-pass caches ----------------------------------------------------------------

    def _restore_preparation_weights(self, prefixes: tuple[str, ...]):
        """Reload only small, previously released tensors when a new pass needs them."""
        names = [k for k in self._released_names if k.startswith(prefixes)]
        if not names:
            return
        from safetensors import safe_open

        with safe_open(str(self.weights_path), framework="numpy") as f:
            for name in names:
                w = mx.array(f.get_tensor(name))
                self.w.p[name] = w.astype(self.dt) if w.dtype == mx.float16 else w
                self._released_names.remove(name)
        mx.eval([self.w.p[k] for k in names])

    def prepare_pass(self, cond: np.ndarray, times) -> list:
        """Evaluate exact per-request conditioning, then release its large weights.

        Each timestep uses the original single-row GEMMs, preserving rounding.
        Cache keys are the actual FP32 sigma bytes, never an approximate time grid.
        A later schedule or image restores only the tensors it needs. No converted
        checkpoint or persistent input cache is created.
        """
        time_prefixes = ("t_embedder.", "adaLN_modulation.")
        self._restore_preparation_weights(time_prefixes)
        self._schedule_cache.clear()
        cached = {}
        for t in times:
            key = np.float32(t).tobytes()
            if key not in cached:
                cached[key] = self.embed(float(np.float32(t)))
                mx.eval(cached[key])
        self._schedule_cache = cached
        if self._prepared_condition is None or not np.array_equal(cond, self._prepared_condition):
            self._restore_preparation_weights(tuple(f"blocks.{i}.cross_attn.to_kv." for i in range(self.num_blocks)))
            self._prepared_kv = self.context_kv(mx.array(cond))
            self._prepared_condition = np.array(cond, copy=True)
        # The negative constant uses to_kv.bias, so compute it before releasing.
        self.negative_cross()
        released = 0
        for name in list(self.w.p):
            if name.startswith(time_prefixes) or ".cross_attn.to_kv." in name:
                released += self.w.p.pop(name).nbytes
                self._released_names.add(name)
        self.preparation_stats = {"cached_times": len(cached), "released_weight_bytes": released,
                                  "cached_time_bytes": sum(a.nbytes for pair in cached.values() for a in pair)}
        mx.clear_cache()
        return self._prepared_kv

    def context_kv(self, cond: mx.array) -> list[tuple[mx.array, mx.array]]:
        """Per-block cross-attention keys/values ``[1, H, L, D]`` for a condition."""
        self._restore_preparation_weights(tuple(f"blocks.{i}.cross_attn.to_kv." for i in range(self.num_blocks)))
        ctx = cond.astype(self.dt)[0]
        kvs = []
        for i in range(self.num_blocks):
            kv = self.w.linear(ctx, f"blocks.{i}.cross_attn.to_kv").astype(self.dt)
            kv = kv.reshape(-1, 2, self.heads, self.head_dim)
            k = kv[:, 0].transpose(1, 0, 2)[None]
            v = kv[:, 1].transpose(1, 0, 2)[None]
            kvs.append((k, v))
        mx.eval(kvs)
        return kvs

    def negative_cross(self) -> list[mx.array]:
        """``to_out(v_bias)`` per block: cross-attention output for a zero condition."""
        if self._negative_cross is None:
            out = []
            for i in range(self.num_blocks):
                if f"blocks.{i}.negative_cross" in self.w.p:
                    out.append(self.w.p[f"blocks.{i}.negative_cross"])
                    continue
                bias = self.w.p[f"blocks.{i}.cross_attn.to_kv.bias"]
                v_bias = bias[self.channels :].astype(self.dt)[None]
                out.append(self.w.linear(v_bias, f"blocks.{i}.cross_attn.to_out").astype(self.dt))
            mx.eval(out)
            self._negative_cross = out
        return self._negative_cross

    # -- building blocks ----------------------------------------------------------------

    def _emb_scale_shift(self, emb: mx.array, name: str) -> tuple[mx.array, mx.array]:
        y = self.w.linear(silu(emb), f"{name}.emb_layers.1").astype(self.dt)
        scale, shift = mx.split(y, 2, axis=1)
        return scale, shift

    def _resblock_tail(self, h: mx.array, name: str, level: Level, scale, shift) -> mx.array:
        """``norm2 -> modulate -> SiLU -> conv2 -> SiLU -> linear`` of a res block."""
        h = layer_norm32(h, eps=1e-6) * (1 + scale) + shift
        h = silu(h)
        h = self.w.conv(h, level, f"{name}.conv2.conv").astype(self.dt)
        h = silu(h)
        return self.w.linear(h, f"{name}.linear").astype(self.dt)

    def _resblock(self, x: mx.array, emb: mx.array, name: str, level: Level) -> mx.array:
        scale, shift = self._emb_scale_shift(emb, name)
        h = layer_norm32(x, self.w.p[f"{name}.norm1.weight"], self.w.p[f"{name}.norm1.bias"])
        h = silu(h)
        h = self.w.conv(h, level, f"{name}.conv1.conv").astype(self.dt)
        h = self._resblock_tail(h, name, level, scale, shift)
        skip = (
            self.w.linear(x, f"{name}.skip_connection").astype(self.dt)
            if f"{name}.skip_connection.bias" in self.w.p
            else x
        )
        return h.astype(self.rt) + skip.astype(self.rt)

    def _down_block(self, x: mx.array, emb: mx.array, geo: Geometry) -> mx.array:
        name = "input_blocks.1"
        scale, shift = self._emb_scale_shift(emb, name)
        x = geo.downsample(x)
        h = layer_norm32(x, self.w.p[f"{name}.norm1.weight"], self.w.p[f"{name}.norm1.bias"])
        h = silu(h)
        h = self.w.conv(h, geo.low, f"{name}.conv1.conv").astype(self.dt)
        h = self._resblock_tail(h, name, geo.low, scale, shift)
        return h.astype(self.rt) + self.w.linear(x, f"{name}.skip_connection").astype(self.rt)

    def _up_conv(self, h_low: mx.array, geo: Geometry, name: str) -> mx.array:
        """Convolution of the upsampled ``gather(h_low)`` without materialising it.

        ``out[o] = sum_k W_k h_low[parent(o + d_k)]``: project every parent through all
        27 taps once (``[N_low, 27, Cout]``) and gather-sum per fine voxel.
        """
        if f"{name}.weight" in self.w.q:
            w = self.w.dense(f"{name}.weight").astype(self.dt)
        else:
            w = self.w.p[f"{name}.weight"]
        cout = w.shape[0]
        cin = h_low.shape[1]
        # [Cout, 27*Cin] -> [Cin, 27*Cout] with column index k*Cout + o.
        w_taps = w.reshape(cout, 27, cin).transpose(2, 1, 0).reshape(cin, 27 * cout)
        proj = (h_low.astype(w.dtype) @ w_taps).reshape(-1, cout)
        proj = mx.concatenate([proj, mx.zeros((1, cout), dtype=proj.dtype)], 0)
        rows = max(1, sparse.GATHER_ELEMENTS // (27 * cout))
        outs = []
        chunked = rows < geo.full.n
        for s in range(0, geo.full.n, rows):
            e = min(s + rows, geo.full.n)
            g = mx.take(proj, geo.up_conv_index[s:e], axis=0).astype(F32)
            outs.append(mx.sum(g, axis=1))
            if chunked:
                mx.eval(outs[-1])
        y = outs[0] if len(outs) == 1 else mx.concatenate(outs, 0)
        return (y + self.w.p[f"{name}.bias"].astype(F32)).astype(self.dt)

    def _up_block(self, x_low: mx.array, emb: mx.array, geo: Geometry) -> mx.array:
        name = "out_blocks.0"
        scale, shift = self._emb_scale_shift(emb, name)
        h = layer_norm32(x_low, self.w.p[f"{name}.norm1.weight"], self.w.p[f"{name}.norm1.bias"])
        h = silu(h)
        h = self._up_conv(h, geo, f"{name}.conv1.conv")
        h = self._resblock_tail(h, name, geo.full, scale, shift)
        skip = geo.upsample(self.w.linear(x_low, f"{name}.skip_connection").astype(self.rt))
        return h.astype(self.rt) + skip

    def _block_fn(self, n_pos: int, has_negative: bool, shared_input: bool = False,
                  shared_attention: bool = False):
        """Compiled ``ModulatedSparseTransformerCrossBlock`` (shared by all blocks).

        ``x`` is ``[B, N, C]`` with the ``n_pos`` conditioned branches first and, when
        ``has_negative``, the zero-condition branch last.
        """
        key = (n_pos, has_negative, shared_input, shared_attention)
        if key in self._block_fns:
            return self._block_fns[key]
        dt, heads, head_dim = self.dt, self.heads, self.head_dim
        qmeta = self._block_qmeta
        scale = head_dim**-0.5

        def lin(x, W, name):
            if hasattr(self, "block_linear"):
                return self.block_linear(x, W, name)
            if self.precision.gemm == "w8a8":
                from . import w8a8

                return w8a8.linear(x, W[f"{name}.weight"], W[f"{name}.w8_scale"], W[f"{name}.bias"])
            b = W[f"{name}.bias"]
            if f"{name}.weight" in qmeta:
                wq, sc, bi = W[f"{name}.weight"]
                group, bits = qmeta[f"{name}.weight"]
                y = mx.quantized_matmul(x.astype(sc.dtype), wq, sc, bi, transpose=True,
                                        group_size=group, bits=bits)
                return y + b.astype(y.dtype)
            w = W[f"{name}.weight"]
            return mx.addmm(b.astype(w.dtype), x.astype(w.dtype), w.T)

        sdpa = lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        flash = dt == mx.float16 and self.precision.attention == "flash"
        fused_qkv = dt in (mx.float16, mx.float32) and head_dim % 2 == 0

        def self_attention_inputs(qkv, W, cos, sin, batch, n):
            """RoPE + QK-RMSNorm and the ``[B, H, N, D]`` layout SDPA wants."""
            if fused_qkv:
                qkv_fn = kernels.qkv_rope_rms_simd if self.precision.optimized and not self.precision.preserve_fp16 else kernels.qkv_rope_rms
                return qkv_fn(
                    qkv.reshape(batch * n, -1), cos, sin, W["self_attn.q_rms_norm.gamma"],
                    W["self_attn.k_rms_norm.gamma"], batch, heads, head_dim)
            qkv = qkv.reshape(batch * n, 3, heads, head_dim)
            cos, sin = mx.tile(cos, (batch, 1)), mx.tile(sin, (batch, 1))
            q = rms_norm_heads(apply_rope(qkv[:, 0], cos, sin), W["self_attn.q_rms_norm.gamma"], dt)
            k = rms_norm_heads(apply_rope(qkv[:, 1], cos, sin), W["self_attn.k_rms_norm.gamma"], dt)
            to_bhnd = lambda t: t.reshape(batch, n, heads, head_dim).transpose(0, 2, 1, 3)
            return to_bhnd(q), to_bhnd(k), to_bhnd(qkv[:, 2])

        def norm(x, scale=None, shift=None, weight=None, bias=None):
            """``LayerNorm32`` (+ adaLN modulation); one fused kernel for fp16 rows."""
            if x.dtype == mx.float16:
                return kernels.layer_norm_modulate(x, 1e-6, weight, bias, scale, shift)
            h = layer_norm32(x, weight, bias)
            return h if scale is None else h * scale + shift

        def block(W, x, mod, cos, sin, ck, cv, negative):
            batch, n, c = x.shape
            parts = mx.split(W["modulation"] + mod.astype(F32), 6, axis=1)
            shift_msa, gate_msa, shift_mlp, gate_mlp = (parts[j].astype(dt) for j in (0, 2, 3, 5))
            # upstream rounds (1 + scale) to the torso dtype when broadcasting it
            scale_msa, scale_mlp = ((1 + parts[j]).astype(dt) for j in (1, 4))

            # Both guidance branches enter block 0 with the same input. They
            # diverge only at cross-attention, so share its self-attention work.
            self_x = x[:1] if shared_input else x
            self_batch = 1 if shared_input else batch
            h = norm(self_x, scale_msa, shift_msa)
            qkv = lin(h, W, "self_attn.to_qkv").astype(dt)
            if shared_attention:
                # Keep the QKV GEMM's original batch/row shape and rounding;
                # both rows are identical, so only one attention is needed.
                qkv = qkv[:1]
                self_batch = 1
            if flash:  # writes [B, N, C] directly
                a = attention.flash_attention(*self_attention_inputs(qkv, W, cos, sin, self_batch, n), scale)
            else:
                a = sdpa(*self_attention_inputs(qkv, W, cos, sin, self_batch, n))
                a = a.transpose(0, 2, 1, 3).reshape(self_batch, n, c)
            if shared_attention:
                # Contiguous duplication also keeps the output GEMM's original
                # shape/layout, avoiding a different matrix-kernel selection.
                a = mx.contiguous(mx.broadcast_to(a, (batch, n, c)))
            x = x + lin(a, W, "self_attn.to_out").astype(dt) * gate_msa

            h = norm(x[:n_pos] if self.precision.optimized and not self.precision.preserve_fp16 and n_pos else x,
                     weight=W["norm2.weight"], bias=W["norm2.bias"])
            cross = []
            for b in range(n_pos):
                qc = lin(h[b], W, "cross_attn.to_q").astype(dt).reshape(1, n, heads, head_dim)
                if flash:
                    o = attention.flash_attention(qc, ck, cv, scale, q_layout="bnhd")[0]
                else:
                    o = sdpa(qc.transpose(0, 2, 1, 3), ck, cv)[0].transpose(1, 0, 2).reshape(n, c)
                cross.append(lin(o, W, "cross_attn.to_out").astype(dt))
            if has_negative:
                cross.append(mx.broadcast_to(negative, (n, c)))
            x = x + mx.stack(cross, 0)

            h = norm(x, scale_mlp, shift_mlp)
            h = gelu_tanh(lin(h, W, "mlp.mlp.0").astype(dt))
            return x + lin(h, W, "mlp.mlp.2").astype(dt) * gate_mlp

        fn = mx.compile(block) if self.compile else block
        self._block_fns[key] = fn
        return fn

    def run_block(self, i: int, x, mod, cos, sin, kv, negative, shared_input=False, shared_attention=False):
        """Apply block ``i``; ``kv`` = ``(k, v)`` of the condition or ``None``."""
        n_pos = 0 if kv is None else 1
        has_negative = negative is not None
        fn = self._block_fn(n_pos, has_negative, shared_input, shared_attention)
        dummy = mx.zeros((1,), dtype=self.dt)
        ck, cv = kv if kv is not None else (dummy, dummy)
        return fn(self.block_weights[i], x, mod, cos, sin, ck, cv,
                  negative if negative is not None else dummy)

    # -- forward ------------------------------------------------------------------------

    def embed(self, t: float) -> tuple[mx.array, mx.array]:
        """Timestep embedding (fp32) and the shared adaLN modulation."""
        key = np.float32(t).tobytes()
        if key in self._schedule_cache:
            return self._schedule_cache[key]
        if self._released_names:
            self._restore_preparation_weights(("t_embedder.", "adaLN_modulation."))
        temb = timestep_embedding(mx.array(np.array([np.float32(1000.0) * np.float32(t)], dtype=np.float32)))
        emb = self.w.linear(temb, "t_embedder.mlp.0")
        emb = self.w.linear(silu(emb), "t_embedder.mlp.2")
        mod = self.w.linear(silu(emb), "adaLN_modulation.1")
        return emb, mod

    def input_stage(self, feats: mx.array, emb: mx.array, geo: Geometry):
        conv_emb = emb.astype(self.dt)
        h = self.w.linear(feats.astype(F32), "input_layer").astype(self.rt)
        h = self._resblock(h, conv_emb, "input_blocks.0", geo.full)
        skip0 = h
        h = self._down_block(h, conv_emb, geo)
        return h, skip0

    def output_stage(self, h: mx.array, skip_low: mx.array, skip0: mx.array, emb, geo) -> mx.array:
        conv_emb = emb.astype(self.dt)
        h = mx.concatenate([h, skip_low], axis=1)
        h = self._up_block(h, conv_emb, geo)
        h = mx.concatenate([h, skip0], axis=1)
        h = self._resblock(h, conv_emb, "out_blocks.1", geo.full)
        h = mx.fast.layer_norm(h.astype(F32), None, None, 1e-5)
        return self.w.linear(h, "out_layer")

    def __call__(self, feats: mx.array, t: float, geo: Geometry, kvs: list) -> list[mx.array]:
        """Velocity for each branch in ``kvs`` (``None`` = zero condition).

        Args:
            feats: ``[N, 32]`` fp32 sample on ``geo.coords``.
            t: Model time ``sigma`` (the network sees ``1000 * t``).
            geo: Pass geometry.
            kvs: Per-branch cached cross-attention K/V lists, or ``None`` for the
                all-zero negative condition.
        """
        emb, mod = self.embed(t)
        mod = mod.astype(self.dt)  # upstream: adaLN_modulation(emb).to(self.dtype)
        h, skip0 = self.input_stage(feats, emb, geo)
        skip_low = h
        batch = len(kvs)
        x = mx.broadcast_to(h[None], (batch, *h.shape))
        positive = [kv for kv in kvs if kv is not None]
        if len(positive) > 1 or (kvs[0] is None and len(kvs) > 1):
            raise ValueError("branches must be [condition] , [condition, None] or [None]")
        neg = self.negative_cross() if len(positive) < len(kvs) else None
        for i in range(self.num_blocks):
            kv_i = positive[0][i] if positive else None
            args = (i, x, mod, geo.rope_cos, geo.rope_sin, kv_i, None if neg is None else neg[i])
            if i == 0 and batch == 2 and self.precision.optimized:
                if self.precision.preserve_fp16 and self.precision.attention == "flash" and self.dt == mx.float16:
                    x = self.run_block(*args, shared_attention=True)
                elif not self.precision.preserve_fp16:
                    x = self.run_block(*args, shared_input=True)
                else:
                    x = self.run_block(*args)
            else:
                x = self.run_block(*args)
        return [self.output_stage(x[b], skip_low, skip0, emb, geo) for b in range(batch)]
