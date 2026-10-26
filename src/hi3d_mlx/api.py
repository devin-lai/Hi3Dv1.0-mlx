"""Image + coarse mesh to refined geometry using released checkpoints."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def refine(image, coarse_mesh, *, output=None, preset="fp16", resolution=512,
           seed=0, steps=None, model_dir=None, local_files_only=False,
           attention="auto", progress=print):
    """Return a trimesh mesh. Use a __main__ guard in scripts (spawned encoders).

    Model downloads are limited to the preset's weights and requested resolution.
    ``model_dir`` uses a preassembled local layout and never downloads checkpoints.
    ``local_files_only`` also prevents encoder downloads. Output is geometry only.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Hi3D MLX requires native arm64 Python on Apple silicon macOS")
    from hi3d_mac.presets import PRESETS
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
    if resolution not in (512, 1024, 1536):
        raise ValueError("Resolution must be 512, 1024 or 1536")
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or steps < 1):
        raise ValueError("steps must be a positive integer")
    if attention not in ("auto", "flash", "mlx"):
        raise ValueError("attention must be auto, flash or mlx")
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError("A Metal GPU is required")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    is_m5 = "Apple M5" in chip
    selected = PRESETS[preset]
    if selected.gemm == "w8a8" and not is_m5:
        raise ValueError("Experimental W8A8 kernels require the validated M5 family")
    if attention == "flash" and not is_m5:
        raise ValueError("The custom flash kernel requires M5; use --attention mlx")
    from PIL import Image
    import trimesh
    from sparc3d.sparcubes.postprocess import validate_mesh
    # Validate inputs before starting multi-gigabyte downloads.
    prepared_image = Image.open(image) if isinstance(image, (str, Path)) else image
    if not isinstance(prepared_image, Image.Image):
        raise TypeError("image must be a PIL image or image path")
    mesh = trimesh.load(coarse_mesh, process=False) if isinstance(coarse_mesh, (str, Path)) else coarse_mesh
    mesh = validate_mesh(mesh)
    from .hub import prepare_models, prepare_encoders
    variant = selected.weights or "fp16"
    progress(f"Preparing {variant} checkpoints for {resolution}³")
    root = prepare_models(variant, resolution, model_dir=model_dir, local_files_only=local_files_only)
    prepare_encoders(local_files_only=local_files_only)
    from dataclasses import replace
    kernel = ("flash" if is_m5 else "mlx") if attention == "auto" else attention
    config = replace(selected, attention=kernel).run_config()
    config.weights_dir = None if variant == "fp16" else root / "models" / variant
    config.capture_intermediates = False
    if steps is not None:
        config.steps = steps
    from hi3d_mac.metrics import Profiler
    from hi3d_mac.mlx_backend.pipeline import Hi3DPipelineMLX
    profiler = Profiler()
    started = time.perf_counter()
    try:
        with profiler.phase("load"):
            pipe = Hi3DPipelineMLX(config, root=root, profiler=profiler, progress=progress)
        result = pipe(prepared_image, mesh, resolution=resolution, seed=seed)
        elapsed = time.perf_counter() - started
    finally:
        profiler.close()
    if output is not None:
        target = Path(output).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        result.export(target)
        metrics = {"preset": preset, "variant": variant, "resolution": resolution,
                   "seed": seed, "steps_per_pass": config.steps, "attention": kernel,
                   "seconds_excluding_download_and_export": elapsed,
                   "hardware": {"chip": chip, "macos": platform.mac_ver()[0],
                                "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))},
                   "environment": {"python": platform.python_version(), "mlx": mx.__version__},
                   "output": {"vertices": len(result.vertices), "faces": len(result.faces)},
                   **profiler.summary()}
        target.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return result
