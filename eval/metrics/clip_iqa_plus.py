"""CLIP-IQA+ (CIQA) no-reference perceptual quality metric. Coding assistance by Cursor Composer 2.5."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from eval.metrics._backend import get_metric, pil_to_tensor, pyiqa_available
from eval.utils import to_rgb

_METRIC: Any = None
_DEVICE: str | None = None


def _load_metric(device: str) -> Any:
    global _METRIC, _DEVICE
    if _METRIC is None or _DEVICE != device:
        if not pyiqa_available():
            raise ImportError("pyiqa is required for CLIP-IQA+ scoring.")
        _METRIC = get_metric("clipiqa+", device=device)
        _DEVICE = device
    return _METRIC


def compute_clip_iqa_plus(
    views: list[Image.Image],
    device: str,
) -> dict:
    """
    Score each view with pyiqa clipiqa+.

    Returns:
        per_view_ciqa: list of float
        mean_ciqa: float
    """
    metric = _load_metric(device)
    per_view_ciqa: list[float] = []
    with torch.no_grad():
        for view in views:
            image_t = pil_to_tensor(to_rgb(view), device)
            score = metric(image_t)
            if score.ndim > 0:
                score = score.squeeze()
            per_view_ciqa.append(float(score.item()))

    mean_ciqa = float(sum(per_view_ciqa) / len(per_view_ciqa)) if per_view_ciqa else float("nan")
    return {"per_view_ciqa": per_view_ciqa, "mean_ciqa": mean_ciqa}


def preload(device: str = "cpu") -> None:
    """Eagerly load the CLIP-IQA+ metric."""
    _load_metric(device)


def unload() -> None:
    """Release cached metric."""
    global _METRIC, _DEVICE
    _METRIC = None
    _DEVICE = None
