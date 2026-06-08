"""CLIP Consistency (CC) metric — image-image cosine similarity vs central view."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from eval.utils import to_rgb

_CLIP_MODEL: Any = None
_CLIP_PREPROCESS: Any = None


def encode_image(
    clip_model: Any,
    clip_preprocess: Any,
    image: Image.Image,
    device: str,
) -> torch.Tensor:
    """Return L2-normalised CLIP image embedding."""
    image_rgb = to_rgb(image)
    with torch.no_grad():
        image_t = clip_preprocess(image_rgb).unsqueeze(0).to(device)
        features = clip_model.encode_image(image_t)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0)


def compute_clip_consistency(
    views: list[Image.Image],
    central_index: int,
    clip_model: Any,
    clip_preprocess: Any,
    device: str,
) -> dict:
    """
    Compute cosine similarity between each view and the central view embedding.

    Returns:
        per_view_cc: list of float (NaN for central view)
        mean_cc: float
    """
    if not views:
        return {"per_view_cc": [], "mean_cc": float("nan")}

    embeddings = [
        encode_image(clip_model, clip_preprocess, view, device) for view in views
    ]
    central_emb = embeddings[central_index]

    per_view_cc: list[float] = []
    valid_scores: list[float] = []
    for idx, emb in enumerate(embeddings):
        if idx == central_index:
            per_view_cc.append(float("nan"))
            continue
        score = float(torch.dot(emb, central_emb).item())
        per_view_cc.append(score)
        valid_scores.append(score)

    mean_cc = float(sum(valid_scores) / len(valid_scores)) if valid_scores else float("nan")
    return {"per_view_cc": per_view_cc, "mean_cc": mean_cc}


def preload(device: str = "cpu") -> tuple[Any, Any]:
    """Load ViT-L/14 via open_clip for CC scoring."""
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        _CLIP_MODEL = model.to(device).eval()
        _CLIP_PREPROCESS = preprocess
    return _CLIP_MODEL, _CLIP_PREPROCESS


def unload() -> None:
    """Release cached CLIP model."""
    global _CLIP_MODEL, _CLIP_PREPROCESS
    _CLIP_MODEL = None
    _CLIP_PREPROCESS = None
