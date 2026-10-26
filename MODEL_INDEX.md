# Model and preset selection

The public inference runtime is in the [main project repository](https://huggingface.co/devin-lai/Hi3Dv1.0-mlx). Install it once, then choose a preset. Checkpoint conversion tooling is not included.

| Variant | Both transformers + shared VAE | Main presets | Tradeoff |
|---|---:|---|---|
| [FP16](https://huggingface.co/devin-lai/Hi3Dv1.0-mlx-fp16) | 7.20 GiB | fp16, lean, fp16-vae16, lean-vae16, steps15, steps10 | Reference weights; lower decoder precision and fewer steps change outputs |
| [INT8](https://huggingface.co/devin-lai/Hi3Dv1.0-mlx-int8) | 4.22 GiB | int8, turbo | Smaller weights; turbo also uses 10 steps and FP16 VAE |
| [INT4](https://huggingface.co/devin-lai/Hi3Dv1.0-mlx-int4) | 2.63 GiB | int4, minimal | Substantial geometry drift; minimal also uses 8 steps and FP16 VAE |
| [W8A8](https://huggingface.co/devin-lai/Hi3Dv1.0-mlx-w8a8) | 4.41 GiB | w8a8, w8a8-turbo | Experimental native integer compute; M5 required |

Sizes describe checkpoint storage, not runtime memory, and exclude the two external image encoders. For 512³, only transformer_512 and the VAE are needed. For 1024³ and 1536³, both transformers are needed. There is no separate 1536 transformer. Shared configs are tiny and reused.

`fp32` uses the original checkpoints with FP32 compute; it does not restore precision already absent from the stored FP16 tensors. `fast`, `fast-vae16`, `fast-cache` and `lean-cache` are experimental arithmetic/cache choices using the FP16 checkpoints. Profiles are described in [models/index.json](models/index.json).

## Index conventions

The catalog links each variant to its custom index. Paths in these indexes are relative to the **assembled model root**, regardless of the JSON file's location. The Hub layout spans repositories: `shared_components` resolve from `devin-lai/Hi3Dv1.0-mlx-shared`, and transformers resolve from the selected precision repository. The CLI assembles that layout automatically and verifies the recorded hashes.

The root `model_index.json` is the original upstream pipeline descriptor, provided by the shared repository. It is not the MLX catalog. The installable runtime's `src/hi3d_mlx/artifacts.json` is the versioned download manifest. Use the provided Hi3D MLX runtime; generic `mlx-lm` loaders do not implement the architecture.

## Quality and measurement scope

Historical evidence covers three objects on Apple M5 / 32 GiB / MLX 0.32.4. Not every profile has a measured run at every resolution. FP16 multi-pass results can diverge from free-running FP32 through changed voxel support and noise assignment; 1536³ has no local FP32 baseline. [Benchmarks and methodology](BENCHMARKS.md) preserve these distinctions.
