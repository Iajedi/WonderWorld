"""Shared pyiqa metric backend for the evaluation environment. Coding assistance by Cursor Composer 2.5."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from eval.utils import composite_for_masked_metric, mask_bounding_box, to_rgb

_METRICS: dict[str, Any] = {}


def pyiqa_available() -> bool:
    try:
        import pyiqa  # noqa: F401

        return True
    except ImportError:
        return False


def get_metric(name: str, device: str = "cpu", **kwargs: Any) -> Any:
    """Lazy-load and cache a pyiqa metric."""
    key = f"{name}:{device}:{sorted(kwargs.items())}"
    if key not in _METRICS:
        import pyiqa

        _METRICS[key] = pyiqa.create_metric(name, device=device, **kwargs)
    return _METRICS[key]


def clear_metric_cache() -> None:
    """Release cached pyiqa metric models."""
    _METRICS.clear()


def preload_pyiqa_metrics(device: str = "cpu") -> None:
    """Eagerly load pyiqa full-reference metrics used by the benchmark."""
    if not pyiqa_available():
        return
    for name in ("psnr", "lpips"):
        get_metric(name, device=device)


def pil_to_tensor(image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Convert RGB PIL image to NCHW tensor in [0, 1]."""
    arr = np.asarray(to_rgb(image), dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def prepare_pair(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image | None,
    *,
    masked_only: bool,
    strategy: str = "composite",
) -> tuple[Image.Image, Image.Image]:
    """Prepare reference/prediction pair for pyiqa full-reference metrics."""
    ref_rgb = to_rgb(ref)
    pred_rgb = to_rgb(pred)
    if pred_rgb.size != ref_rgb.size:
        pred_rgb = pred_rgb.resize(ref_rgb.size, Image.Resampling.LANCZOS)

    if not masked_only:
        return ref_rgb, pred_rgb

    if mask is None:
        raise ValueError("mask is required when masked_only=True")

    if strategy == "bbox":
        box = mask_bounding_box(mask)
        return ref_rgb.crop(box), pred_rgb.crop(box)

    if strategy == "composite":
        return ref_rgb, composite_for_masked_metric(ref_rgb, pred_rgb, mask)

    raise ValueError(f"Unknown prepare_pair strategy: {strategy}")


def score_fr_metric(
    metric_name: str,
    ref: Image.Image,
    pred: Image.Image,
    *,
    device: str = "cpu",
    metric_kwargs: dict[str, Any] | None = None,
) -> float:
    """Run a pyiqa full-reference metric on an image pair."""
    metric = get_metric(metric_name, device=device, **(metric_kwargs or {}))
    ref_t = pil_to_tensor(ref, device)
    pred_t = pil_to_tensor(pred, device)
    with torch.no_grad():
        score = metric(pred_t, ref_t)
    return float(score.item())
