"""MLX refinement pipeline with the control flow of ``Sparc3DPipeline``.

Model evaluations (transformer, VAE decoder) run in MLX. Everything that fixes the
numerics of the trajectory is reused from upstream: ``prepare_image``,
``ImageConditioner`` (fp32), the CPU ``torch.Generator`` noise, the
``FlowMatchHeunScheduler`` sigmas, guidance constants, ``adaptive_vox`` (NumPy
port), FlexiCubes extraction, hole filling and postprocessing.
"""

from __future__ import annotations

import gc
import hashlib
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
import torch

from hi3d_mac import paths
from hi3d_mac.torch_backend import install

from .transformer import Geometry, Precision, Sparc3DTransformerMLX
from .vae import DecodedPoints, SparconvVAEDecoderMLX

STRUCTURE_RESOLUTIONS = (64, 128, 192)
OUTPUT_RESOLUTIONS = (512, 1024, 1536)
PASS_BUDGETS = (80_000, 90_000, 110_000)
TARGET_FACE_COUNTS = (1_000_000, 1_000_000, 2_000_000)
DECODE_FEATURE_CEILING = 12_000_000
GUIDANCE_START = 0.7


@dataclass
class RunConfig:
    """Numerics and schedule of an MLX run."""

    transformer: Precision = field(default_factory=Precision)
    vae_dtype: mx.Dtype = mx.float32
    steps: int = 25
    cache_limit_gb: float = 1.0
    """MLX buffer-cache bound; freed buffers above it go back to the OS."""
    weights_dir: Path | None = None
    """Directory with converted (quantized) transformer files; ``None`` = official fp16."""
    scheduler: str = "heun"
    """``heun`` (released default), ``euler`` (upstream's first-order alternative) or
    ``ab2`` (second-order Adams-Bashforth: one evaluation per step, reusing the
    previous slope; an approximation of the Heun trajectory)."""
    pass_steps: tuple[int, ...] | None = None
    """Per-pass step counts overriding ``steps`` (e.g. ``(25, 12, 12)``)."""
    forced_coords: dict[int, np.ndarray] | None = None
    """Validation only: voxel coordinates to use for these pass indices instead of
    voxelising the previous pass's mesh (e.g. the FP32 baseline's, so a later pass
    starts from the same voxels and therefore the same noise rows)."""
    corrector_guidance: str = "full"
    """``full``: the Heun corrector slope is guided like upstream. ``reuse``: the
    corrector reuses the predictor's guidance difference (pos - neg) instead of
    evaluating the zero-condition branch again (approximation)."""
    lean_extract: bool = True
    """Use :mod:`hi3d_mac.lean_extract` for mesh extraction (same mesh, less memory)."""
    capture_intermediates: bool = True
    """Retain numerical validation arrays; disable for production / memory benchmarks."""
    serial_conditioner: bool = False
    """Load DINO and CLIP in separate sequential child processes to reduce encoding peak."""
    residual_cache_threshold: float = 0.0
    """Opt-in late transformer-tail reuse; zero disables this approximation."""
    stable_prefix: bool = False
    """Use original FP16/FP32-VAE arithmetic before the final pass, preserving its
    voxel support/noise assignment. Load the VAE only when needed."""


def guidance_scale(t: float) -> float:
    """``effective_guidance_scale`` evaluated as upstream does, in float32."""
    t32 = np.float32(t)
    if t32 >= 1.0:
        return 4.0
    if t32 >= GUIDANCE_START:
        return float(np.float32(10.0) * (t32 * t32))
    return 1.0


class Hi3DPipelineMLX:
    def __init__(
        self,
        config: RunConfig | None = None,
        root: Path = paths.MODEL_ROOT,
        profiler: Any | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        config = RunConfig() if config is None else config
        install()
        from hi3d_mac import fast_unique, lean_extract

        fast_unique.install()  # exact GPU row-unique for FlexiCubes / hole filling
        if config.lean_extract:
            lean_extract.install()  # packed-key corner / edge tables, bitwise-identical meshes
        mx.set_cache_limit(int(config.cache_limit_gb * 1e9))
        self.config = config
        self.root = root
        self.profiler = profiler
        self.progress = progress or (lambda message: None)
        self._transformers: dict[str, Sparc3DTransformerMLX] = {}
        self.vae = None if config.stable_prefix else SparconvVAEDecoderMLX(root / "vae", dtype=config.vae_dtype)
        from sparc3d.pipelines import loader

        self.scheduler_config = loader._read_json(root / "scheduler" / "scheduler_config.json")
        self._loader = loader
        self.captured: dict = {"passes": []}

    # -- components ---------------------------------------------------------------------

    @contextmanager
    def _phase(self, name: str):
        """Profiler phase plus the MLX allocator's own peak inside it."""
        mx.reset_peak_memory()
        with self.profiler.phase(name) if self.profiler else nullcontext():
            yield
        self.captured.setdefault("mlx_memory", []).append({
            "phase": name,
            "mlx_peak_gb": mx.get_peak_memory() / 2**30,
            "mlx_active_gb": mx.get_active_memory() / 2**30,
            "mlx_cache_gb": mx.get_cache_memory() / 2**30,
        })

    def transformer(self, name: str, stable: bool = False) -> Sparc3DTransformerMLX:
        """Load one transformer, releasing the other (only one pass runs at a time)."""
        precision = replace(self.config.transformer, optimized=False) if stable else self.config.transformer
        if name not in self._transformers or self._transformers[name].precision != precision:
            self._transformers.clear()
            gc.collect()
            mx.clear_cache()
            weights_file = None
            if self.config.weights_dir is not None:
                weights_file = self.config.weights_dir / f"{name}.safetensors"
            cls, extra = Sparc3DTransformerMLX, {}
            if self.config.residual_cache_threshold and not stable:
                from .residual_cache import CachedTransformer

                cls, extra = CachedTransformer, {"threshold": self.config.residual_cache_threshold}
            self._transformers[name] = cls(self.root / name, precision,
                                           weights_file=weights_file, **extra)
        return self._transformers[name]

    def condition(self, prepared_image) -> np.ndarray:
        """Official fp32 ``ImageConditioner`` on MPS, in a child process."""
        from hi3d_mac import conditioner_process

        fn = conditioner_process.encode_serial if self.config.serial_conditioner else conditioner_process.encode
        return fn(prepared_image)

    # -- stages ---------------------------------------------------------------------------

    def sample(self, model, cond, coords, structure, generator, pass_index) -> np.ndarray:
        from sparc3d.schedulers.flow_match_euler import FlowMatchEulerScheduler
        from sparc3d.schedulers.flow_match_heun import FlowMatchHeunScheduler

        steps = self.config.steps
        if hasattr(model, "reset_cache"):
            model.reset_cache()
        if self.config.pass_steps is not None:
            steps = self.config.pass_steps[min(pass_index, len(self.config.pass_steps) - 1)]
        n = coords.shape[0]
        noise = torch.randn((n, 32), generator=generator, device="cpu", dtype=torch.float32)
        cls = FlowMatchHeunScheduler if self.config.scheduler == "heun" else FlowMatchEulerScheduler
        # every scheduler shares the token-dependent sigma schedule
        scheduler = cls(**{k: v for k, v in self.scheduler_config.items() if k != "_class_name"})
        timesteps = scheduler.set_timesteps(steps, num_tokens=n)
        sigmas = scheduler.sigmas.numpy().astype(np.float32)
        sigma_hats = timesteps.numpy().astype(np.float32)

        geo = Geometry(coords)
        if self.config.transformer.optimized:
            times = np.concatenate([sigma_hats, sigmas[1:]]) if self.config.scheduler == "heun" else sigma_hats
            kv = model.prepare_pass(cond, times)
            self.captured.setdefault("conditioning_preparation", []).append(dict(model.preparation_stats))
        else:
            kv = model.context_kv(mx.array(cond))
        x = mx.array(noise.numpy())

        last_delta = None

        def velocity(sample: mx.array, t: np.float32, reuse: bool = False) -> mx.array:
            nonlocal last_delta
            if t < GUIDANCE_START:
                (v,) = model(sample, float(t), geo, [kv])
                return v
            if reuse and last_delta is not None:
                # v = neg + s (pos - neg) = pos + (s - 1)(pos - neg), with the
                # guidance difference taken from the predictor evaluation.
                (pos,) = model(sample, float(t), geo, [kv])
                return pos + (guidance_scale(t) - 1.0) * last_delta
            pos, neg = model(sample, float(t), geo, [kv, None])
            last_delta = pos - neg
            return neg + guidance_scale(t) * (pos - neg)

        reuse = self.config.corrector_guidance == "reuse"
        previous = None  # (slope, model time) of the last step, for ab2
        for i in range(steps):
            t, t_next = sigma_hats[i], sigmas[i + 1]
            dt = float(np.float32(t - t_next))
            last_delta = None
            v1 = velocity(x, t)
            if self.config.scheduler == "heun":
                predicted = x - dt * v1
                v2 = velocity(predicted, t_next, reuse=reuse)
                x = x - dt * (0.5 * (v1 + v2))
            elif self.config.scheduler == "ab2" and previous is not None:
                # Adams-Bashforth 2 on the non-uniform grid of evaluation times:
                # extrapolate the slope to the middle of this step.
                v0, t0 = previous
                ratio = dt / (2.0 * float(np.float32(t0 - t)))
                x = x - dt * (v1 + ratio * (v1 - v0))
            else:
                x = x - dt * v1
            previous = (v1, t)
            mx.eval(x)
            self.progress(f"pass {pass_index + 1} step {i + 1}/{steps}")
        latents = x * model.latent_std + model.latent_mean
        mx.eval(latents)
        if hasattr(model, "cache_events"):
            self.captured.setdefault("residual_cache", []).append({
                "pass": pass_index, "threshold": self.config.residual_cache_threshold,
                "evaluations": model.cache_calls, "hits": sum(e["reuse"] for e in model.cache_events),
                "events": model.cache_events})
        del kv, geo
        return np.array(latents)

    def decode(self, coords, latents, structure, dtype=None) -> DecodedPoints:
        dtype = self.config.vae_dtype if dtype is None else dtype
        if self.vae is None or self.vae.dt != dtype:
            self.vae = None
            gc.collect()
            mx.clear_cache()
            self.vae = SparconvVAEDecoderMLX(self.root / "vae", dtype=dtype)
        return self.vae(coords, latents, structure)

    @staticmethod
    def extract(decoded: DecodedPoints):
        """Official ``point_to_mesh`` + ``fill_mesh_holes`` on CPU tensors.

        With :mod:`hi3d_mac.fast_unique` installed (the pipeline does this) the
        row-unique steps run on the GPU; the output is bitwise identical.
        """
        import trimesh
        from sparc3d.sparcubes.extract import point_to_mesh
        from sparc3d.sparcubes.fill_hole import fill_mesh_holes
        from sparc3d.structures.point import Point

        layouts = {}
        start = 0
        for key in ("sdf", "deform"):
            entry = Point.ALL_LAYOUTS[key]
            layouts[key] = {**entry, "range": (start, start + entry["size"])}
            start += entry["size"]
        n = decoded.grid_coord.shape[0]
        point = Point(
            {
                "feat": torch.from_numpy(decoded.feat),
                "grid_coord": torch.from_numpy(decoded.grid_coord.astype(np.int64)),
                "sdf_sign": torch.from_numpy(decoded.sdf_sign),
                "batch": torch.zeros(n, dtype=torch.long),
                "resolution": decoded.resolution,
                "matrix": torch.from_numpy(decoded.matrix),
            }
        )
        point["layouts"] = layouts
        mesh = fill_mesh_holes(point_to_mesh(point))
        if not mesh.success:
            raise RuntimeError("mesh extraction produced an empty isosurface")
        return trimesh.Trimesh(
            vertices=mesh.vertices.numpy(), faces=mesh.faces.numpy(), process=False
        )

    # -- driver ---------------------------------------------------------------------------

    def __call__(self, image, coarse_mesh, resolution: int = 1024, seed: int = 0):
        from sparc3d.ops import adaptive_vox
        from sparc3d.sparcubes.postprocess import postprocess_mesh, validate_mesh
        from sparc3d.utils.image import prepare_image

        target = OUTPUT_RESOLUTIONS.index(resolution)
        # A pipeline can serve several images. Captures belong to one request;
        # keeping old arrays both leaks memory and corrupts per-run diagnostics.
        self.captured = {"passes": []}
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        prepared = prepare_image(image)
        mesh = validate_mesh(coarse_mesh)
        with self._phase("condition"):
            cond = self.condition(prepared)
        if self.config.capture_intermediates:
            self.captured["condition"] = cond
        with self._phase("voxelize"):
            coords = adaptive_vox(mesh, resolution=64, target_num=PASS_BUDGETS[0]).numpy()
        for index in range(target + 1):
            structure = STRUCTURE_RESOLUTIONS[index]
            stable = self.config.stable_prefix and index < target
            vae_dtype = mx.float32 if stable else self.config.vae_dtype
            record = {"voxels": int(coords.shape[0]),
                      "coords_sha256": hashlib.sha256(np.ascontiguousarray(coords, dtype=np.int32).tobytes()).hexdigest(),
                      "stable_prefix_pass": stable, "vae_dtype": str(vae_dtype)}
            if self.config.capture_intermediates:
                record["coords"] = coords
            self.captured["passes"].append(record)
            name = "transformer_512" if structure == 64 else "transformer_1024"
            with self._phase("load"):
                model = self.transformer(name, stable=stable)
            with self._phase("sample"):
                latents = self.sample(model, cond, coords, structure, generator, index)
            record["latents_sha256"] = hashlib.sha256(np.ascontiguousarray(latents, dtype=np.float32).tobytes()).hexdigest()
            if self.config.capture_intermediates:
                record["latents"] = latents
            del model  # the cache below must hold the only reference
            if index == target or structure == 64:
                # transformer_512 only serves the 64³ pass, and after the last pass no
                # transformer is needed: free it before decoding.
                self._transformers.clear()
                gc.collect()
                mx.clear_cache()
            with self._phase("decode"):
                decoded = self.decode(coords, latents, structure, dtype=vae_dtype)
            if self.config.stable_prefix:
                self.vae = None  # decoder weights need not overlap extraction or the next sampler
                gc.collect()
            mx.clear_cache()
            record["decoded_points"] = int(decoded.grid_coord.shape[0])
            if index < target and decoded.grid_coord.shape[0] > DECODE_FEATURE_CEILING:
                raise RuntimeError("decoded feature ceiling exceeded before another pass")
            with self._phase("extract"):
                mesh = self.extract(decoded)
            del decoded
            record["mesh_face_count"] = int(len(mesh.faces))
            if self.config.capture_intermediates:
                record["mesh_vertices"] = np.asarray(mesh.vertices, np.float32)
                record["mesh_faces"] = np.asarray(mesh.faces, np.int64)
            if index < target:
                forced = (self.config.forced_coords or {}).get(index + 1)
                with self._phase("voxelize"):
                    coords = forced if forced is not None else adaptive_vox(
                        mesh,
                        resolution=STRUCTURE_RESOLUTIONS[index + 1],
                        target_num=PASS_BUDGETS[index + 1],
                    ).numpy()
        with self._phase("postprocess"):
            return postprocess_mesh(mesh, target_face_count=TARGET_FACE_COUNTS[target])
