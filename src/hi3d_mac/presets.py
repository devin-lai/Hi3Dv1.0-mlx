"""Named MLX run configurations used by the scripts and the reports."""

from __future__ import annotations

from dataclasses import dataclass, field

from hi3d_mac import paths


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    transformer_dtype: str = "float16"
    vae_dtype: str = "float32"
    steps: int = 25
    pass_steps: tuple[int, ...] | None = None
    """Per-pass step counts (64³, 128³, 192³ passes) overriding ``steps``."""
    scheduler: str = "heun"
    corrector_guidance: str = "full"
    weights: str | None = None
    """Subdirectory of the external ``HI3D_CONVERTED_ROOT`` with converted transformer files (quantized)."""
    attention: str = "flash"
    """fp16 attention kernel: ``flash`` (neural-accelerator kernel, :mod:`hi3d_mac.mlx_backend.attention`)
    or ``mlx`` (``mx.fast.scaled_dot_product_attention``)."""
    gemm: str = "mlx"
    optimized: bool = False
    preserve_fp16: bool = False
    residual_cache_threshold: float = 0.0
    stable_prefix: bool = False
    env: dict = field(default_factory=lambda: {"MLX_ENABLE_TF32": "0"})
    """MLX fp32 matmuls default to TF32 on M5; every preset keeps true FP32 where
    it computes in float32 (VAE, fp32 tensors), so TF32 is off by default."""

    def run_config(self):
        import mlx.core as mx

        from hi3d_mac.mlx_backend.pipeline import RunConfig
        from hi3d_mac.mlx_backend.transformer import Precision

        return RunConfig(
            transformer=Precision(dtype=getattr(mx, self.transformer_dtype), attention=self.attention,
                                  gemm=self.gemm, optimized=self.optimized, preserve_fp16=self.preserve_fp16),
            vae_dtype=getattr(mx, self.vae_dtype),
            steps=self.steps,
            pass_steps=self.pass_steps,
            scheduler=self.scheduler,
            corrector_guidance=self.corrector_guidance,
            weights_dir=None if self.weights is None else paths.CONVERTED_DIR / self.weights,
            residual_cache_threshold=self.residual_cache_threshold,
            stable_prefix=self.stable_prefix,
        )

    def describe(self) -> dict:
        return {
            "description": self.description,
            "transformer_dtype": self.transformer_dtype,
            "weights": self.weights or "official fp16 checkpoint",
            "vae_dtype": self.vae_dtype,
            "attention": self.attention if self.transformer_dtype == "float16" else "mlx",
            "gemm": self.gemm,
            "optimized": self.optimized,
            "preserve_fp16": self.preserve_fp16,
            "residual_cache_threshold": self.residual_cache_threshold,
            "stable_prefix": self.stable_prefix,
            "scheduler": self.scheduler,
            "pass_steps": self.pass_steps,
            "corrector_guidance": self.corrector_guidance,
            "env": self.env,
        }


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        # exact / near-identical (task 2)
        Preset("fp32", "exact FP32 everywhere (TF32 off); reproduces the FP32 baseline",
               transformer_dtype="float32"),
        Preset("fp16", "released precision: fp16 transformer torso, FP32 VAE; the default"),
        Preset("lean", "original FP16 arithmetic and FP32 VAE; exact conditioning preparation and shorter weight residency",
               optimized=True, preserve_fp16=True, stable_prefix=True),
        Preset("lean-vae16", "lean with final-pass FP16 decoder; original transformer arithmetic and stable preliminary passes",
               optimized=True, preserve_fp16=True, stable_prefix=True, vae_dtype="float16"),
        Preset("lean-cache", "experimental lean-vae16 with bounded 1.2% late residual reuse; inspect gallery",
               optimized=True, preserve_fp16=True, stable_prefix=True, vae_dtype="float16", residual_cache_threshold=.012),
        Preset("fast", "25 Heun steps, FP16 torso, coalesced QKV, shared first self-attention, released conditioning weights",
               optimized=True, stable_prefix=True),
        Preset("fast-vae16", "fast with FP16 VAE GEMMs; inspect multi-pass voxel sensitivity",
               optimized=True, vae_dtype="float16", stable_prefix=True),
        Preset("fast-cache", "experimental fast-vae16 with bounded late residual reuse (1.2% threshold); inspect gallery",
               optimized=True, vae_dtype="float16", residual_cache_threshold=.012, stable_prefix=True),
        # approximations (task 3)
        Preset("int8", "fp16 compute with 8-bit transformer-block weights", weights="int8"),
        Preset("int4", "fp16 compute with 4-bit transformer-block weights", weights="int4"),
        Preset("steps15", "fp16, 15 Heun steps per pass instead of 25", steps=15),
        Preset("steps10", "fp16, 10 Heun steps per pass instead of 25", steps=10),
        Preset("turbo", "10 Heun steps, 8-bit weights, fp16 VAE", steps=10, weights="int8",
               vae_dtype="float16"),
        Preset("minimal", "8 Heun steps, 4-bit weights, fp16 VAE: fastest and smallest", steps=8,
               weights="int4", vae_dtype="float16"),
        Preset("fp16-vae16", "fp16 transformer and fp16 VAE GEMMs (official CUDA autocast precision)",
               vae_dtype="float16"),
        Preset("w8a8", "experimental signed-H64 native int8 transformer GEMMs, 25 steps; inspect comparison stacks",
               weights="w8a8", gemm="w8a8"),
        Preset("w8a8-turbo", "experimental signed-H64 native int8 GEMMs, 10 steps, fp16 VAE",
               weights="w8a8", gemm="w8a8", steps=10, vae_dtype="float16"),
    )
}
"""Sampler alternatives (Euler, Adams-Bashforth, corrector guidance reuse) lost to
plain fewer Heun steps in ``scripts/research/sampler_study.py``; they remain
available through :class:`RunConfig` but not as presets."""
