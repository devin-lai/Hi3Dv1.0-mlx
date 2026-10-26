"""Run the official fp32 image conditioner in a short-lived child process.

PyTorch-MPS returns freed buffers to its driver but the process footprint keeps
them until macOS reclaims the pages. Encoding the image in a spawned child keeps
the ~3 GB of DINOv2-large + CLIP-L weights out of the MLX process entirely; the
child exits once the ``[1, 2917, 2048]`` tokens are returned. The computation is
the upstream ``ImageConditioner`` on MPS in fp32, unchanged.
"""

from __future__ import annotations

import io
import multiprocessing as mp

import numpy as np


def _encode(png: bytes) -> np.ndarray:
    from hi3d_mac import paths
    import torch
    from PIL import Image

    from hi3d_mac.torch_backend import install

    install()
    from sparc3d.pipelines import loader

    conditioner = loader.load_conditioner(
        cache_dir=paths.HF_HUB_CACHE, local_files_only=True,
        dino_model_id=paths.encoder_directory(paths.DINO_MODEL_ID),
        clip_model_id=paths.encoder_directory(paths.CLIP_MODEL_ID),
    ).to("mps")
    with torch.inference_mode():
        return conditioner(Image.open(io.BytesIO(png))).float().cpu().numpy()


def encode(prepared_image) -> np.ndarray:
    """Tokens of an already ``prepare_image``-framed PIL image."""
    buffer = io.BytesIO()
    prepared_image.save(buffer, format="PNG")  # lossless: the child sees identical pixels
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_encode, (buffer.getvalue(),))


def _encode_component(png: bytes, component: str) -> np.ndarray:
    """One official FP32 encoder, with the upstream preprocessing and normalization.

    Runs in its own process so all weights and MPS driver buffers are reclaimed
    before the other encoder starts. This trades a second process startup for
    a lower memory floor. Compare token arrays before using this path.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModel, CLIPVisionModel
    from hi3d_mac import paths
    from hi3d_mac.torch_backend import install

    install()
    from sparc3d.models.conditioners.image import ImageConditioner, _DINO_SIZE, _CLIP_SIZE

    image = Image.open(io.BytesIO(png))
    device = torch.device("mps")
    kwargs = {"cache_dir": str(paths.HF_HUB_CACHE), "local_files_only": True}
    with torch.inference_mode(), torch.autocast(device_type="mps", enabled=False):
        if component == "dino":
            model = AutoModel.from_pretrained(paths.encoder_directory(paths.DINO_MODEL_ID), **kwargs).float().to(device).eval()
            x = F.interpolate(ImageConditioner._pil_tensor(image, device), size=(_DINO_SIZE, _DINO_SIZE),
                              mode="bilinear", align_corners=False)
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif component == "clip":
            model = CLIPVisionModel.from_pretrained(paths.encoder_directory(paths.CLIP_MODEL_ID), **kwargs).float().to(device).eval()
            x = ImageConditioner._image_tensor(image, _CLIP_SIZE, Image.Resampling.LANCZOS, device)
            mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        else:
            raise ValueError(component)
        # Match upstream: buffers are formed as CPU float32 then transferred.
        mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        out = model(pixel_values=(x-mean)/std)
        tokens = out.last_hidden_state if component == "dino" else out.pooler_output
        return F.normalize(tokens.float(), p=2, dim=-1).cpu().numpy()


def encode_serial(prepared_image) -> np.ndarray:
    """Same conditioning tokens, with encoder weights resident one at a time."""
    buffer = io.BytesIO()
    prepared_image.save(buffer, format="PNG")
    # Separate pool lifetimes guarantee complete exit before the second model
    # loads; no persistent parent-side torch/MPS encoders are introduced.
    outputs = []
    for component in ("dino", "clip"):
        with mp.get_context("spawn").Pool(1) as pool:
            outputs.append(pool.apply(_encode_component, (buffer.getvalue(), component)))
    dino, clip = outputs
    if dino.shape != (1, 2917, 1024) or clip.shape != (1, 1024):
        raise ValueError(f"unexpected conditioner shapes: {dino.shape}, {clip.shape}")
    return np.concatenate([dino, np.broadcast_to(clip[:, None], dino.shape)], axis=-1)
