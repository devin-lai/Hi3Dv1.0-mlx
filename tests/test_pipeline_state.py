"""Request isolation contracts for a reusable inference pipeline."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hi3d_mac.mlx_backend import pipeline


def test_default_config_is_not_shared(monkeypatch, tmp_path):
    from hi3d_mac import fast_unique, lean_extract
    pipeline.install()
    from sparc3d.pipelines import loader
    monkeypatch.setattr(fast_unique, "install", lambda: None)
    monkeypatch.setattr(lean_extract, "install", lambda: None)
    monkeypatch.setattr(pipeline, "SparconvVAEDecoderMLX", lambda *a, **kw: None)
    monkeypatch.setattr(loader, "_read_json", lambda p: {})
    first = pipeline.Hi3DPipelineMLX(root=tmp_path)
    first.config.steps = 8
    second = pipeline.Hi3DPipelineMLX(root=tmp_path)
    assert second.config.steps == 25
    assert first.config is not second.config


def test_reused_pipeline_records_only_current_request(monkeypatch):
    pipeline.install()
    import sparc3d.ops
    import sparc3d.sparcubes.postprocess as post
    import sparc3d.utils.image as image_utils
    monkeypatch.setattr(image_utils, "prepare_image", lambda x: x)
    monkeypatch.setattr(post, "validate_mesh", lambda x: x)
    monkeypatch.setattr(post, "postprocess_mesh", lambda x, **kw: x)
    monkeypatch.setattr(sparc3d.ops, "adaptive_vox", lambda *a, **kw: torch.zeros((1, 3), dtype=torch.int32))
    pipe = pipeline.Hi3DPipelineMLX.__new__(pipeline.Hi3DPipelineMLX)
    pipe.config = pipeline.RunConfig(capture_intermediates=False)
    pipe.captured = {"passes": []}
    pipe.profiler = None
    pipe._transformers = {}
    pipe.condition = lambda image: np.zeros((1, 1, 1), np.float32)
    pipe.transformer = lambda *a, **kw: object()
    pipe.sample = lambda *a: np.zeros((1, 32), np.float32)
    pipe.decode = lambda *a, **kw: SimpleNamespace(grid_coord=np.zeros((1, 3)))
    mesh = SimpleNamespace(faces=np.zeros((1, 3)), vertices=np.zeros((3, 3)))
    pipe.extract = lambda decoded: mesh
    pipe(None, mesh, resolution=512)
    first = pipe.captured
    pipe(None, mesh, resolution=512)
    assert len(pipe.captured["passes"]) == 1
    assert len(pipe.captured["mlx_memory"]) == len(first["mlx_memory"])
    assert pipe.captured is not first
