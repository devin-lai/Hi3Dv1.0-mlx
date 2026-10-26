# Inference guide

Follow the [quick start](README.md#quick-start). There is no PyPI release; clone this repository and run `uv pip install .` from its root in a Python 3.12 environment. Only inference and pre-converted checkpoint loading are included.

## Hardware

The historical benchmark used Apple M5, 32 GiB RAM, macOS 27.0.1, Python 3.12.13 and MLX 0.32.4. Release smoke tests used an Apple M5 Pro with 64 GiB and Python 3.12.15. A 32 GiB M5-class machine is the measured starting point; 8/16 GiB configurations are not validated. Background apps, image encoders and extraction buffers add to memory usage.

The CLI selects custom flash attention on M5. On earlier Apple-silicon generations it selects MLX's native attention; this fallback is not covered by the end-to-end hardware measurements. Experimental W8A8 requires M5 and is rejected on other chips. Intel Macs, Linux and CPU-only inference are unsupported by this package. Use `--attention mlx` to explicitly select native MLX attention.

## Inputs and outputs

Both a reference image and a coarse triangle mesh of the same object are required. Match orientation and framing. The model refines the supplied coarse shape; it does not perform background removal or generate the initial mesh. Transparent inputs are handled by the upstream image preparation.

Outputs contain geometry only and are centered and normalized to unit maximum radius. The sparse extraction-grid setting (512³/1024³/1536³) is not a fixed triangle count. GLB is the recommended output format. The CLI refuses to overwrite an existing output file.

## Presets

```bash
hi3d-mlx refine --image object.png --coarse-mesh coarse.glb \
  --preset int8 --resolution 512 --seed 0 --output refined.glb
```

Start with `fp16` for quality, or `int8` for smaller weights. `int4` and `minimal` have substantial geometric drift in some cases. `lean` uses the original FP16 arithmetic with shorter weight residency. `lean-vae16` reduces final decoder precision. `fast`, `fast-vae16`, `fast-cache`, `lean-cache` and W8A8 are experimental. Reduced-step presets are approximate. Full choices: `hi3d-mlx refine --help` and [MODEL_INDEX.md](MODEL_INDEX.md).

## Selective downloads and offline use

```bash
hi3d-mlx download --variant int8 --resolution 512
hi3d-mlx refine --image object.png --coarse-mesh coarse.glb \
  --preset int8 --resolution 512 --output refined.glb --local-files-only
```

A 512³ run downloads one transformer and the shared VAE/configuration, plus the pinned DINOv2 and CLIP image encoders. A 1024³ or 1536³ run needs both transformers. Switching between 1024³ and 1536³ does not download another transformer. The results dataset is never downloaded by inference.

The default model layout is `~/.cache/hi3d-mlx/models`, with links to the standard Hugging Face cache. Set `HI3D_CACHE` to relocate the assembled model layout and `HF_HUB_CACHE` to relocate the Hub cache. `HF_HOME` is also respected. Both encoders are loaded from their exact recorded snapshots, not the moving `main` reference.

For an existing complete local model layout, use `--model-dir /path/to/model`. This verifies the files without downloading checkpoints. Encoders are still required; add `--local-files-only` to forbid their download too. Index paths are relative to the assembled model root. The shared components live in the shared repository; selected transformers live in their precision repository. SHA-256 verification runs before model loading and can take a few seconds for large files.

## Python API

```python
from hi3d_mlx import refine

if __name__ == "__main__":
    mesh = refine(
        "reference.png", "coarse.glb",
        preset="fp16", resolution=512, seed=0,
        output="refined.glb",
    )
```

The `__main__` guard is required because image conditioning uses spawned subprocesses. Run scripts from a file, not an unguarded notebook cell. Set `MallocLargeCache=0 MLX_ENABLE_TF32=0` before starting Python to match the CLI's allocator and FP32 policy. Low-level pipeline objects support repeated serial requests; they are not thread-safe for simultaneous calls.

## Troubleshooting

- **Missing encoder in offline mode:** run `hi3d-mlx download` without `--local-files-only` once. The encoder revisions are pinned.
- **Wrong checkpoint size or checksum:** the selected local file is incomplete or changed. Download the released version again; do not mix another model's checkpoints.
- **Memory pressure:** close other GPU applications, start at 512³, and consider INT8. Checkpoint storage numbers are not RAM requirements.
- **Mesh differs after changing precision or step count:** quantization and sampler changes can alter geometry. At higher resolutions, a changed coarse-pass surface can change voxel support and noise assignment for subsequent passes.
- **Custom attention fails to compile:** try `--attention mlx` for FP16/INT8/INT4 and report the chip, macOS and MLX versions. W8A8 requires the M5-specific kernel path.

## Tests

```bash
uv pip install '.[test]'
MLX_ENABLE_TF32=0 python -m pytest -q
```

GPU tests require Apple silicon; W8A8 and custom flash tests require M5. The inference package contains no checkpoint-conversion entry point.
