"""Peak Signal-to-Noise Ratio metric."""

from __future__ import annotations

import numpy as np
from PIL import Image

from eval.metrics._backend import prepare_pair, pyiqa_available, score_fr_metric
from eval.utils import masked_pixel_arrays, to_rgb


def _compute_numpy(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image | None,
    *,
    masked_only: bool,
) -> float:
    ref_rgb = to_rgb(ref)
    pred_rgb = to_rgb(pred)
    if ref_rgb.size != pred_rgb.size:
        pred_rgb = pred_rgb.resize(ref_rgb.size, Image.Resampling.LANCZOS)

    if masked_only:
        if mask is None:
            raise ValueError("mask is required when masked_only=True")
        ref_pixels, pred_pixels = masked_pixel_arrays(ref_rgb, pred_rgb, mask)
        mse = float(np.mean((ref_pixels - pred_pixels) ** 2))
    else:
        ref_arr = np.array(ref_rgb, dtype=np.float32)
        pred_arr = np.array(pred_rgb, dtype=np.float32)
        mse = float(np.mean((ref_arr - pred_arr) ** 2))

    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def compute(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image | None = None,
    *,
    masked_only: bool = True,
    device: str = "cpu",
) -> float:
    """Compute PSNR between reference and prediction."""
    if masked_only:
        return _compute_numpy(ref, pred, mask, masked_only=True)

    if pyiqa_available():
        ref_rgb, pred_rgb = prepare_pair(ref, pred, mask, masked_only=False)
        return score_fr_metric("psnr", ref_rgb, pred_rgb, device=device)

    return _compute_numpy(ref, pred, mask, masked_only=False)
