# Changelog

## 1.0.0 — 2026-10-26

First public GitHub release of the Hi3D v1.0 MLX inference runtime.

- CLI and Python API for image-and-coarse-mesh geometry refinement.
- Selective, checksum-verified FP16, INT8, INT4 and experimental M5 W8A8 downloads from versioned Hugging Face repositories.
- Shared VAE/configuration cache and pinned DINOv2/CLIP encoder revisions.
- Independent default pipeline configurations, per-request capture resets, and safe handling of empty sparse geometry.
- A runnable dragon example, actual output visuals, documented benchmark data, 47 local regression/numerical tests and four full 512³ format smoke runs.
- GitHub packaging/download CI and contribution templates.

Inference source matches the validated Hugging Face v1.0.0 source inventory. Checkpoint conversion code and conversion research remain private. See [validation](VALIDATION.md) and [upstream modifications](MODIFICATIONS.md) for scope and attribution.
