"""Metric aggregation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from eval.metrics import clip_score, fid_metric, lpips_metric, psnr


METRIC_NAMES = ("psnr", "fid", "lpips", "clip_score")


def aggregate_per_sample(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    masked_only: bool = True,
    device: str = "cpu",
) -> dict[str, float]:
    """Compute per-sample metrics (PSNR, LPIPS, CLIP). FID is batch-level."""
    return compute_per_sample(
        ref,
        pred,
        mask,
        prompt,
        masked_only=masked_only,
        device=device,
    )


def compute_per_sample(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    masked_only: bool = True,
    device: str = "cpu",
) -> dict[str, float]:
    """Compute per-sample metrics excluding batch FID."""
    return {
        "psnr": psnr.compute(ref, pred, mask, masked_only=masked_only, device=device),
        "lpips": lpips_metric.compute(
            ref, pred, mask, masked_only=masked_only, device=device
        ),
        "clip_score": clip_score.compute(
            ref,
            pred,
            mask,
            prompt=prompt,
            device=device,
            masked_only=False,
        ),
    }


def compute_all(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    masked_only: bool = True,
    device: str = "cpu",
    batch_fid: float | None = None,
) -> dict[str, float]:
    """Compute all metrics for one sample; FID must be supplied from batch compute."""
    metrics = compute_per_sample(
        ref, pred, mask, prompt, masked_only=masked_only, device=device
    )
    if batch_fid is not None:
        metrics["fid"] = batch_fid
    return metrics


def preload_models(device: str = "cpu") -> None:
    """Load heavy metric models before batched scoring."""
    from eval.metrics._backend import preload_pyiqa_metrics

    preload_pyiqa_metrics(device)
    fid_metric.preload(device)
    lpips_metric.preload(device)
    clip_score.preload(device)


def unload_models() -> None:
    """Release cached metric models."""
    from eval.metrics._backend import clear_metric_cache

    clear_metric_cache()
    fid_metric.unload()
    lpips_metric.unload()
    clip_score.unload()


def summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return mean/std/min/max for each numeric metric."""
    summary: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        values = [float(r[name]) for r in results if name in r and r[name] is not None]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if name == "fid":
            summary[name] = {
                "value": float(finite[0] if len(finite) else arr[0]),
                "count": float(len(arr)),
            }
            continue
        summary[name] = {
            "mean": float(np.mean(finite if len(finite) else arr)),
            "std": float(np.std(finite if len(finite) else arr)),
            "min": float(np.min(finite if len(finite) else arr)),
            "max": float(np.max(finite if len(finite) else arr)),
            "count": float(len(arr)),
        }
    return summary
