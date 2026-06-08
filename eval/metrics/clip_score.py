"""CLIP prompt adherence metric."""

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


def _load_transformers_clip(device: str) -> tuple[Any, Any, Any]:
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor, None


def _load_clip(device: str) -> tuple[str, Any, Any, Any]:
    global _BACKEND, _MODEL, _PREPROCESS, _TOKENIZER
    if _MODEL is None:
        backend_order = {
            "open_clip": ("open_clip", "pyiqa", "transformers"),
            "pyiqa": ("pyiqa", "open_clip", "transformers"),
            "transformers": ("transformers", "open_clip", "pyiqa"),
        }.get(CLIP_BACKEND, ("open_clip", "pyiqa", "transformers"))

        for backend in backend_order:
            if backend == "pyiqa" and pyiqa_available():
                try:
                    _BACKEND = "pyiqa"
                    _MODEL = _load_pyiqa_clipscore(device)
                    _PREPROCESS = None
                    _TOKENIZER = None
                    return _BACKEND, _MODEL, _PREPROCESS, _TOKENIZER
                except Exception:
                    continue
            if backend == "open_clip":
                try:
                    model, preprocess, tokenizer = _load_open_clip(device)
                    _BACKEND = "open_clip"
                    _MODEL = model
                    _PREPROCESS = preprocess
                    _TOKENIZER = tokenizer
                    return _BACKEND, _MODEL, _PREPROCESS, _TOKENIZER
                except ImportError:
                    continue
            if backend == "transformers":
                try:
                    model, preprocess, tokenizer = _load_transformers_clip(device)
                    _BACKEND = "transformers"
                    _MODEL = model
                    _PREPROCESS = preprocess
                    _TOKENIZER = tokenizer
                    return _BACKEND, _MODEL, _PREPROCESS, _TOKENIZER
                except Exception:
                    continue
        raise RuntimeError("No CLIP backend available (tried open_clip, pyiqa, transformers).")
    return _BACKEND, _MODEL, _PREPROCESS, _TOKENIZER


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
        if backend == "pyiqa":
            image_t = pil_to_tensor(image, device)
            score = model(image_t, caption_list=[prompt])
            if score.ndim > 0:
                score = score.squeeze()
        elif backend == "open_clip":
            image_t = preprocess(image).unsqueeze(0).to(device)
            text_t = tokenizer([prompt]).to(device)
            image_features = model.encode_image(image_t)
            text_features = model.encode_text(text_t)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            score = (image_features @ text_features.T).squeeze()
        else:
            inputs = preprocess(
                text=[prompt],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
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
