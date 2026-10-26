# Contributing

Useful contributions include reproducible bug reports, documented results on additional Apple chips, inference improvements, and clearer examples. Start with an [issue](https://github.com/devin-lai/Hi3Dv1.0-mlx/issues/new/choose) when a proposed change affects model behavior or the public API.

## Development setup

Use native arm64 Python 3.12 on an Apple-silicon Mac with Metal support. From the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[test]'
python -m pytest -q
python scripts/verify_release.py
```

The full numerical suite needs Apple hardware; W8A8 and custom kernels were validated on the M5 family. GitHub Actions runs the hardware-independent downloader tests, source checks and wheel packaging on Linux. It neither downloads the model checkpoints nor benchmarks GPU inference.

## Reporting a result

Include the chip, installed memory, macOS, Python/MLX versions, preset, resolution, seed and steps. Attach the generated metrics JSON after checking it for private paths. Share the input image/coarse mesh only when you have permission to redistribute them. A before/after view and a downloadable mesh are more useful than a speed claim alone.

For timing comparisons, use identical saved inputs and settings, download models beforehand, record concurrent workloads, and report repeated samples rather than only the best run. Label cold and warm starts. Do not compare parent-only memory with process-tree memory. See [BENCHMARKS.md](BENCHMARKS.md) for the existing measurement scope.

## Pull requests

- Describe the user-visible problem, the behavior after the change, and how you checked it.
- Keep inference code, documentation and relevant regression coverage together.
- Preserve the upstream and third-party notices. Contributions must be compatible with the repository's existing licensing terms.
- Keep checkpoints, credentials, generated output directories and conversion code out of Git. Weight conversion tooling is outside this repository's scope.
- When intentionally changing public Python source or tests, refresh `runtime-manifest.json` after reviewing the diff:

```bash
python scripts/verify_release.py --update-manifest
python scripts/verify_release.py
```

The manifest records reviewed inference/test files and their hashes. Updating it is an explicit review step, not evidence that new code has been tested.

Keep discussion respectful and specific. Hardware reports with limitations are welcome even when performance is slower than the published examples.
