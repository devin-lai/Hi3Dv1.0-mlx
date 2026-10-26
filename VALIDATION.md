# Release validation · 2026-10-26

## Bugs fixed

1. **Shared mutable pipeline defaults:** independent pipeline instances previously shared the same default RunConfig object. A change to one instance could alter later instances. Each instance now creates its own default configuration.
2. **Diagnostics retained across requests:** repeated calls on one pipeline appended pass records and retained intermediate arrays from earlier requests. Captures now reset for each request, avoiding stale diagnostics and retained request data.

The request-isolation regressions were reproduced with failing tests before the fixes. The empty-key failure was reproduced with a direct call before adding regression coverage. The corresponding tests pass after the fixes. Original benchmark checkpoint bytes and historical measurements were preserved.

3. **Empty geometry handling:** empty packed-key sets caused an invalid MLX scatter, and empty voxel grids caused a NumPy reduction error. Both now preserve empty-array shapes and dtypes without launching invalid operations.

## Verification

- The original 40 numerical/adapter tests passed before changes. Two request-isolation regression tests were added and passed after the fixes.
- **47 public-package tests passed**, including numerical kernels, sparse operations, request isolation, selective resolution/variant downloads, local-file checksum rejection, pinned encoder revisions and W8A8 integer projection checks.
- All **nine checkpoints** matched their recorded SHA-256 hashes. Safetensors bounds/headers, every floating tensor's finiteness, affine quantization layouts and W8A8 scale shapes were checked. See [weights.json](validation/weights.json).
- Full **25-step 512³** inference completed for **FP16, INT8, INT4 and W8A8** through the public CLI on Apple M5 Pro / 64 GiB / macOS 27.0.1 / Python 3.12.15 / MLX 0.32.4. Output meshes are nonempty, finite, correctly indexed and normalized. See [smoke.json](validation/smoke.json).
- A **1024³ lean two-pass smoke test** completed with two steps per pass. It verifies the multi-pass execution path, not high-resolution quality or full-schedule timing.
- The project built into a wheel, installed successfully, and its installed command and embedded download manifest were checked outside the source import path.
- The release source was selected explicitly and recorded in [runtime-manifest.json](runtime-manifest.json). Checkpoint conversion entry points and weight-quantization helpers are absent. W8A8 activation quantization remains because inference requires it.

## GitHub distribution

The GitHub source inventory matches the validated inference/test source above. Weights remain in versioned Hugging Face repositories. GitHub Actions checks the source inventory, documentation links, wheel contents and seven hardware-independent downloader regressions; it does not run the Metal numerical suite or full model inference. The GitHub repository includes 18 unchanged historical benchmark rows and a chart reproduction script.

## Scope

These checks establish the tested behavior, not absence of all possible bugs. Fresh M5 Pro smoke timings are not substituted into the historical M5 performance charts. Earlier Mac generations and 8/16 GiB machines have not received end-to-end validation. W8A8 remains experimental. The complete archived numerical comparisons and per-run qualifications remain available in the [results dataset](https://huggingface.co/datasets/devin-lai/Hi3Dv1.0-mlx-results).
