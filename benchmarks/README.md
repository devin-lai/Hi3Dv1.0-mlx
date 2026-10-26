# Benchmark evidence

`benchmark-512.json` contains 18 unchanged historical rows: three saved objects across the FP32 reference, FP16, INT8, INT4, Turbo and Minimal configurations. Its provenance includes the versioned source URL, source SHA-256 and measurement scope. `benchmark-512.csv` is a compact view of the same rows; blank quality values mean no score is recorded for the reference itself.

The homepage chart uses only the six FP32/FP16 rows listed in [asset provenance](../assets/provenance.json). All cases were recorded on Apple M5 / 32 GiB, with one run per case/configuration. Different precision and reduced-step profiles are not controlled kernel comparisons. The F-score measures agreement with the local FP32 mesh, not ground truth.

Read [BENCHMARKS.md](../BENCHMARKS.md) before reusing these numbers. The [Hugging Face results dataset](https://huggingface.co/datasets/devin-lai/Hi3Dv1.0-mlx-results) retains all 87 timing observations, meshes, inputs and galleries.

## Recreate the chart

From the repository root, in an environment with Matplotlib:

```bash
uv pip install matplotlib
python scripts/plot_benchmarks.py --output assets/benchmarks.png
```

This plots existing measurements. It does not run inference or manufacture new measurements.
