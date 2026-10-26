"""Download only the selected precision and resolution; verify checkpoint bytes."""
from __future__ import annotations

from importlib.resources import files
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def catalog() -> dict:
    return json.loads(files("hi3d_mlx").joinpath("artifacts.json").read_text())


def required_artifacts(variant: str, resolution: int) -> list[tuple[str, dict]]:
    data = catalog()
    if variant not in data["variants"]:
        raise ValueError(f"Unknown variant {variant!r}")
    if resolution not in (512, 1024, 1536):
        raise ValueError("Resolution must be 512, 1024 or 1536")
    spec = data["variants"][variant]
    artifacts = [(data["shared_repo"], item) for item in spec["shared"]]
    names = ["transformer_512"] + (["transformer_1024"] if resolution > 512 else [])
    return artifacts + [(spec["repo_id"], spec["transformers"][name]) for name in names]


def verify_file(path: Path, spec: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Incorrect model size: {path}")
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    if h.hexdigest() != spec["sha256"]:
        raise ValueError(f"Model checksum mismatch: {path}")


def prepare_models(variant: str, resolution: int, *, model_dir=None, local_files_only=False) -> Path:
    from hi3d_mac import paths
    root = paths.MODEL_ROOT if model_dir is None else Path(model_dir).expanduser().resolve()
    data = catalog()
    for repo_id, item in required_artifacts(variant, resolution):
        destination = root / item["path"]
        if model_dir is None:
            cached = Path(hf_hub_download(repo_id=repo_id, filename=item["path"],
                                         revision=data["release"], cache_dir=paths.HF_HUB_CACHE,
                                         local_files_only=local_files_only))
            verify_file(cached, item)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink() and destination.resolve() == cached.resolve():
                continue
            if destination.exists() and not destination.is_symlink():
                verify_file(destination, item)
                continue
            temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
            try:
                temporary.symlink_to(cached.resolve())
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            verify_file(destination, item)
    return root


def prepare_encoder(model_id: str, *, local_files_only=False) -> Path:
    from hi3d_mac import paths
    spec = next((e for e in catalog()["encoders"] if e["id"] == model_id), None)
    if spec is None:
        raise ValueError(f"Unknown image encoder {model_id!r}")
    return Path(snapshot_download(repo_id=model_id, revision=spec["revision"],
                                  cache_dir=paths.HF_HUB_CACHE, local_files_only=local_files_only,
                                  allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"]))


def prepare_encoders(*, local_files_only=False) -> None:
    for encoder in catalog()["encoders"]:
        prepare_encoder(encoder["id"], local_files_only=local_files_only)
