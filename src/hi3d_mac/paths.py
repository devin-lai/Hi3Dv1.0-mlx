"""Portable runtime locations. No conversion workspace is required."""
from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("HI3D_CACHE", Path.home() / ".cache/hi3d-mlx")).expanduser().resolve()
MODEL_ROOT = Path(os.environ.get("HI3D_MODEL_ROOT", CACHE_DIR / "models")).expanduser().resolve()
CONVERTED_DIR = MODEL_ROOT / "models"
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")).expanduser().resolve()
DINO_MODEL_ID = "facebook/dinov2-large"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


def upstream_on_path() -> None:
    """The attributed sparc3d inference modules are installed with this package."""


def encoder_directory(model_id: str) -> Path:
    from hi3d_mlx.hub import prepare_encoder
    return prepare_encoder(model_id, local_files_only=True)
