"""CLIP prompt adherence metric. Using OpenCLIP for CLIP score"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from eval.config import CLIP_BACKEND
from eval.metrics._backend import pil_to_tensor, pyiqa_available
from eval.utils import mask_bounding_box, to_rgb

_BACKEND: str | None = None
_MODEL: Any = None
_PREPROCESS: Any = None
_TOKENIZER: Any = None


def _load_pyiqa_clipscore(device: str) -> Any:
    from eval.metrics._backend import get_metric

    return get_metric("clipscore", device=device)


def _load_open_clip(device: str) -> tuple[Any, Any, Any]:
    import open_clip

    from eval.config import CLIP_MODEL_NAME, CLIP_PRETRAINED

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def _load_clip(device: str) -> tuple[str, Any, Any, Any]:
    global _BACKEND, _MODEL, _PREPROCESS, _TOKENIZER
    model, preprocess, tokenizer = _load_open_clip(device)
    _BACKEND = "open_clip"
    _MODEL = model
    _PREPROCESS = preprocess
    _TOKENIZER = tokenizer
    return _BACKEND, _MODEL, _PREPROCESS, _TOKENIZER

# Compute cosine similarity between generated image and prompt.
# Use OpenCLIP for CLIP score
# With ViT-L/14 model: https://github.com/mlfoundations/open_clip
def compute(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image | None = None,
    *,
    prompt: str,
    device: str = "cpu",
    masked_only: bool = False,
) -> float:
    """Compute cosine similarity between generated image and prompt."""
    del ref

    backend, model, preprocess, tokenizer = _load_clip(device)
    image = to_rgb(pred)
    if masked_only:
        if mask is None:
            raise ValueError("mask is required when masked_only=True for CLIP score.")
        image = image.crop(mask_bounding_box(mask))

    with torch.no_grad():
        image_t = preprocess(image).unsqueeze(0).to(device)
        text_t = tokenizer([prompt]).to(device)
        image_features = model.encode_image(image_t)
        text_features = model.encode_text(text_t)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = (image_features @ text_features.T).squeeze()

    return float(score.item())


def preload(device: str = "cpu") -> None:
    """Eagerly load the CLIP backend."""
    _load_clip(device)


def unload() -> None:
    """Release cached CLIP models."""
    global _BACKEND, _MODEL, _PREPROCESS, _TOKENIZER
    _BACKEND = None
    _MODEL = None
    _PREPROCESS = None
    _TOKENIZER = None
