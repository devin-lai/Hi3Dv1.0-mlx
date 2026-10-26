from pathlib import Path
import hashlib
import pytest
from hi3d_mlx import hub


@pytest.mark.parametrize("variant", ["fp16", "int8", "int4", "w8a8"])
def test_512_download_never_includes_high_resolution_weights(variant):
    items = hub.required_artifacts(variant, 512)
    weights = [(repo, item) for repo, item in items if item["path"].endswith(".safetensors")]
    assert len(weights) == 2  # VAE and one transformer
    assert all("1024" not in item["path"] for _, item in weights)
    assert {repo for repo, _ in weights} == {hub.catalog()["shared_repo"], f"devin-lai/Hi3Dv1.0-mlx-{variant}"}


def test_1536_reuses_both_transformers():
    assert hub.required_artifacts("int4", 1024) == hub.required_artifacts("int4", 1536)


def test_local_models_never_use_network_and_detect_same_size_corruption(tmp_path, monkeypatch):
    good = b"abcd"
    item = {"path": "weights.safetensors", "bytes": 4, "sha256": hashlib.sha256(good).hexdigest()}
    (tmp_path / item["path"]).write_bytes(good)
    monkeypatch.setattr(hub, "required_artifacts", lambda *a: [("unused", item)])
    monkeypatch.setattr(hub, "hf_hub_download", lambda **kw: pytest.fail("Local models must not trigger downloads"))
    assert hub.prepare_models("fp16", 512, model_dir=tmp_path) == tmp_path
    (tmp_path / item["path"]).write_bytes(b"abce")
    with pytest.raises(ValueError, match="checksum mismatch"):
        hub.prepare_models("fp16", 512, model_dir=tmp_path)


def test_encoder_downloads_use_recorded_revision(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(hub, "snapshot_download", lambda **kw: calls.append(kw) or str(tmp_path))
    for encoder in hub.catalog()["encoders"]:
        assert hub.prepare_encoder(encoder["id"], local_files_only=True) == tmp_path
        assert calls[-1]["revision"] == encoder["revision"]
        assert calls[-1]["local_files_only"] is True
        assert calls[-1]["allow_patterns"] == ["config.json", "model.safetensors", "preprocessor_config.json"]
