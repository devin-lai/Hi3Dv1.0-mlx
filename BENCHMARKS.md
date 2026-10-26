# Benchmarks and measurement scope

## Historical cohort

The homepage uses the original **three 512³ cases**, with one run per configuration: dragon, bust and locomotive. All configurations use the same saved image/coarse-mesh inputs and seed 0. The chart compares MLX FP16 and the local PyTorch-MPS FP32 reference at **25 Heun steps**. It is a precision/backend comparison, not an equal-precision kernel speedup or an NVIDIA comparison.

Recorded machine: **Apple M5, 32 GiB**, macOS 27.0.1, Python 3.12.13, MLX 0.32.4. This is distinct from the M5 Pro machine used for release validation. Timings include model loading, image conditioning, sparse sampling, decoding, extraction and postprocessing; downloads, initial coarse generation and final file export are excluded. Measurements came from a shared machine and are not service guarantees. The three examples do not establish performance across all inputs.

Source: [benchmark.json](https://huggingface.co/datasets/devin-lai/Hi3Dv1.0-mlx-results/blob/main/datasets/measurements/benchmark.json), [benchmark.csv](https://huggingface.co/datasets/devin-lai/Hi3Dv1.0-mlx-results/blob/main/datasets/measurements/benchmark.csv) and [run index](https://huggingface.co/datasets/devin-lai/Hi3Dv1.0-mlx-results/blob/main/datasets/index.json). The chart's exact run IDs are in [assets/provenance.json](assets/provenance.json).

The 18 rows used in this repository's tables are also included as [JSON](benchmarks/benchmark-512.json) and [CSV](benchmarks/benchmark-512.csv), with the versioned source and its SHA-256. [Recreate the chart](benchmarks/README.md#recreate-the-chart) using the included plotting script.

## Precision and quality

Surface metrics compare against the local FP32 reference after centering and unit-maximum-radius normalization. F-score uses a 0.002 distance threshold. Surface sampling and nearest-triangle candidates approximate dense-mesh agreement; this is not a ground-truth reconstruction benchmark. Hidden geometry is inferred from the input and coarse shape.

| 512³ profile | Dragon seconds | Bust seconds | Locomotive seconds | F-score range @ 0.002 |
|---|---:|---:|---:|---:|
| FP16 / 25 steps | 95.3 | 126.6 | 128.4 | 0.9877–0.9996 |
| INT8 / 25 steps | 103.7 | 136.7 | 139.7 | 0.9855–0.9957 |
| INT4 / 25 steps | 99.3 | 131.2 | 134.1 | 0.1935–0.5910 |
| Turbo / INT8 / 10 steps | 46.3 | 57.9 | 62.0 | 0.2121–0.8368 |
| Minimal / INT4 / 8 steps | 37.5 | 46.9 | 48.4 | 0.4339–0.4880 |

Quantized storage is not automatically faster. The fastest profiles change both quantization and step count and can substantially change the mesh. W8A8 and arithmetic/cache experiments remain marked experimental. Use the [offline galleries](https://huggingface.co/datasets/devin-lai/Hi3Dv1.0-mlx-results) to inspect those tradeoffs.

At 1024³/1536³, changes in a preliminary pass can change the next pass's voxel support and assign the same random sequence to different positions. Free-running FP16/FP32 disagreement therefore includes those trajectory changes. There is no local FP32 baseline at 1536³ and no independent NVIDIA oracle.

## Memory

The homepage's historical memory chart uses **parent-process physical footprint in GiB**, measured through macOS `proc_pid_rusage`. It includes Metal allocations charged to the parent but excludes short-lived image-encoder subprocesses. It is not total system RAM or a minimum supported memory configuration.

Later records and the public CLI also report a sampled **process-tree physical footprint** at 50 Hz (child discovery at 5 Hz). Shared pages may be counted more than once. Parent and process-tree columns must not be mixed. Some legacy rows lack process-tree data; missing values are not zero.

## Repeated timing records and screens

The complete dataset retains 87 measured runs and 68 unique mesh files. Duplicate mesh bytes share one file, while their timing observations remain separate. The later [speed summary](https://huggingface.co/datasets/devin-lai/Hi3Dv1.0-mlx-results/blob/main/datasets/measurements/speed-summary.json) records sample counts, medians, ranges and machine load. These repeat measurements are not silently substituted into the original cohort.

`screen_passed: null` means no matching strict screen is recorded. It does not mean a pass. Experiments that fail the strict precision screen remain visible and labeled.

## Fresh release validation

The [release validation](VALIDATION.md) and [machine-readable smoke records](validation/smoke.json) document full 512³ inference through the public runtime for all four weight formats on **Apple M5 Pro / 64 GiB / Python 3.12.15**. They establish successful execution and valid output meshes. They are not a controlled speed comparison: test/upload workloads may overlap, and no matching fresh FP32 baseline was run. Historical charts above retain the original M5 labels.
