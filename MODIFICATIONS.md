# Modifications and release scope

This community adaptation is published by Devin Lai. The original Hi3D Open License Agreement, attribution and third-party notices are retained.

- The original transformer and VAE checkpoint bytes are unchanged in the FP16 and shared repositories.
- The INT8 and INT4 transformer files contain pre-quantized affine block weights with original sensitive tensors retained.
- The experimental W8A8 transformer files contain signed-H64 INT8 projection weights and scales; image-conditioning projections retain their recorded precision.
- `src/hi3d_mac/` is the MLX inference port, Metal inference kernels, Mac adapters and request profiling. Weight-conversion functions have been excluded from this public copy.
- `src/hi3d_mlx/` adds the installable command line interface, selective Hub downloads, digest verification and pinned encoder loading.
- `src/sparc3d/` preserves the attributed upstream inference source snapshot at `23dcc684bc6e1fab86ecafb99ec5270edad122f9` from [Hi3Dv1.0](https://github.com/hi3d-official/Hi3Dv1.0). The Mac adapter replaces CUDA-only operations at runtime.
- Empty packed-key and voxel-grid handling now returns valid empty arrays. Pipeline request isolation was fixed: default configurations are independent, and per-request captures reset before inference.
- Public documentation, repository manifests and visual benchmark summaries were prepared for this release. Historical benchmark observations were not rerun or relabeled as current hardware measurements.

The checkpoint converter, weight-conversion helpers, experimental research scripts and private measurement provenance are not part of the public source release. Runtime activation quantization is included because W8A8 inference requires it.
