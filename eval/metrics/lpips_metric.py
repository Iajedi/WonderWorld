"""Learned Perceptual Image Patch Similarity metric. Coding assistance by Cursor Composer 2.5."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from eval.config import LPIPS_NET
from eval.metrics._backend import prepare_pair, pyiqa_available, score_fr_metric
from eval.utils import composite_for_masked_metric, to_rgb

_MODEL: Any = None


def _get_lpips_fallback(device: str) -> Any:
    global _MODEL
    if _MODEL is None:
        import lpips

        _MODEL = lpips.LPIPS(net=LPIPS_NET).to(device)
        _MODEL.eval()
    return _MODEL


def _pil_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(to_rgb(image), dtype=np.float32).copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def _compute_lpips_fallback(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image | None,
    *,
    masked_only: bool,
    device: str,
) -> float:
    ref_rgb = to_rgb(ref)
    pred_rgb = to_rgb(pred)
    if pred_rgb.size != ref_rgb.size:
        pred_rgb = pred_rgb.resize(ref_rgb.size, Image.Resampling.LANCZOS)

    if masked_only:
        if mask is None:
            raise ValueError("mask is required when masked_only=True")
        pred_rgb = composite_for_masked_metric(ref_rgb, pred_rgb, mask)

    model = _get_lpips_fallback(device)
    with torch.no_grad():
        ref_t = _pil_to_tensor(ref_rgb, device)
        pred_t = _pil_to_tensor(pred_rgb, device)
        score = model(ref_t, pred_t)
    return float(score.item())


def compute(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image | None = None,
    *,
    masked_only: bool = True,
    device: str = "cpu",
) -> float:
    """Compute LPIPS distance (lower is better)."""
    if pyiqa_available():
        ref_rgb, pred_rgb = prepare_pair(
            ref,
            pred,
            mask,
            masked_only=masked_only,
            strategy="composite",
        )
        return score_fr_metric(
            "lpips",
            ref_rgb,
            pred_rgb,
            device=device,
        )

    return _compute_lpips_fallback(
        ref, pred, mask, masked_only=masked_only, device=device
    )


def preload(device: str = "cpu") -> None:
    """Eagerly load the LPIPS backend."""
    if pyiqa_available():
        from eval.metrics._backend import get_metric

        get_metric("lpips", device=device)
    else:
        _get_lpips_fallback(device)


def unload() -> None:
    """Release cached LPIPS models."""
    global _MODEL
    _MODEL = None
