"""Run the unmodified official ``sparc3d`` package on Apple silicon.

The official code imports three CUDA-only libraries. :func:`install` registers
exact replacements before ``sparc3d`` is imported:

* ``spconv.pytorch``  -> :mod:`.sparse_conv` (gather/GEMM submanifold convolution)
* ``flash_attn``      -> :mod:`.attention` (exact attention via PyTorch SDPA)
* Triton voxelizer    -> :mod:`.voxelize` (NumPy port of the same kernels)

Everything else (transformer, VAE decoder, schedulers, FlexiCubes, hole filling,
postprocessing, pipeline) is the upstream code at tag v1.0.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
import warnings

_INSTALLED = False


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    # A spec keeps importlib.util.find_spec() working for libraries that probe
    # optional packages (transformers does); the shims have no dist metadata, so
    # such probes still report the real package as not installed.
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    module.__hi3d_mac_shim__ = True
    return module


def _register_spconv() -> None:
    from . import sparse_conv

    pytorch = _module("spconv.pytorch")
    for name in ("SubMConv3d", "SparseConvTensor", "ConvAlgo"):
        setattr(pytorch, name, getattr(sparse_conv, name))
    ops = _module("spconv.pytorch.ops")
    ops.get_indice_pairs = sparse_conv.get_indice_pairs
    core = _module("spconv.pytorch.core")
    core.IndiceData = sparse_conv.IndiceData
    pytorch.ops = ops
    pytorch.core = core
    root = _module("spconv")
    root.pytorch = pytorch
    sys.modules.update(
        {
            "spconv": root,
            "spconv.pytorch": pytorch,
            "spconv.pytorch.ops": ops,
            "spconv.pytorch.core": core,
        }
    )


def _register_flash_attn() -> None:
    from . import attention

    module = _module("flash_attn")
    module.flash_attn_varlen_qkvpacked_func = attention.flash_attn_varlen_qkvpacked_func
    module.flash_attn_varlen_kvpacked_func = attention.flash_attn_varlen_kvpacked_func
    sys.modules["flash_attn"] = module


def _fp32_serialized_attention(blocks) -> None:
    """Keep VAE window attention in float32.

    Upstream casts ``qkv`` to half only because flash-attn has no float32 kernel;
    the FP32 reference evaluates the same attention without that rounding.
    """

    def forward(self, point):
        heads, channels = self.num_heads, self.channels
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat)[order]
        feat = self._flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv.reshape(-1, 3, heads, channels // heads),
            cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=0,
            softmax_scale=self.scale,
        ).reshape(-1, channels)
        feat = feat.to(qkv.dtype)[inverse]
        point.feat = self.proj(feat)
        return point

    blocks.SerializedAttention.forward = forward


def install(*, fp32_vae_attention: bool = True) -> None:
    """Register the Mac backends and import the official package.

    Args:
        fp32_vae_attention: Evaluate the VAE window attention in float32 instead of
            upstream's half-precision flash-attn input cast.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    from hi3d_mac.paths import upstream_on_path

    for name in ("spconv", "flash_attn"):
        existing = sys.modules.get(name)
        if existing is not None and not getattr(existing, "__hi3d_mac_shim__", False):
            raise RuntimeError(f"a real {name} is already imported; refusing to shadow it")
    # Resolve transformers' optional-dependency probes before the shims exist.
    import transformers.utils.import_utils  # noqa: F401

    _register_spconv()
    _register_flash_attn()
    upstream_on_path()
    # Upstream wraps float32 islands in autocast("cuda", enabled=False), which only
    # warns when CUDA is absent.
    warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
    warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")

    import sparc3d.ops as ops
    import sparc3d.ops.voxelize as voxelize
    from sparc3d.models.autoencoders import blocks

    from . import voxelize as mac_voxelize

    mac_voxelize.install(voxelize)
    ops.MeshVoxelizer = voxelize.MeshVoxelizer
    ops.adaptive_vox = voxelize.adaptive_vox
    if fp32_vae_attention:
        _fp32_serialized_attention(blocks)
    _INSTALLED = True
