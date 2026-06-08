"""Metric aggregation for geometric editing (GeoBench) evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from eval.data.geobench_geometry import EDIT_PARAM_TYPE_NAMES, classify_edit_param_type
from eval.metrics import clip_score, lpips_metric, psnr
from eval.utils import read_csv_dicts, to_grayscale, write_csv_dicts

GEOMETRIC_METRIC_NAMES = ("psnr", "fid", "lpips", "clip_score", "clip_score_whole")
EDIT_TYPE_METRIC_NAMES = ("psnr", "lpips", "clip_score")

_REF_METRICS_WARNING_EMITTED = False


def _emit_ref_warning() -> None:
    global _REF_METRICS_WARNING_EMITTED
    if not _REF_METRICS_WARNING_EMITTED:
        from eval.logging_utils import get_logger

        get_logger(__name__).warning(
            "No reference edited image in manifest; skipping PSNR/LPIPS/FID. "
            "Computing CLIP only."
        )
        _REF_METRICS_WARNING_EMITTED = True


def reset_ref_warning() -> None:
    """Reset the one-time reference-metrics warning (for tests)."""
    global _REF_METRICS_WARNING_EMITTED
    _REF_METRICS_WARNING_EMITTED = False


def compute_geometric_per_sample(
    pred: Image.Image,
    prompt: str,
    *,
    ref: Image.Image | None = None,
    mask: Image.Image | None = None,
    masked_only: bool = True,
    device: str = "cpu",
) -> dict[str, float]:
    """Compute per-sample metrics for geometric editing.

    When ``ref`` is absent, PSNR and LPIPS are returned as NaN and only CLIP
    is computed against the generated image and prompt.
    """
    metrics: dict[str, float] = {
        "clip_score_whole": clip_score.compute(
            pred,
            pred,
            mask=None,
            prompt=prompt,
            device=device,
            masked_only=False,
        ),
    }
    if mask is not None:
        metrics["clip_score"] = clip_score.compute(
            pred,
            pred,
            mask=mask,
            prompt=prompt,
            device=device,
            masked_only=True,
        )
    else:
        metrics["clip_score"] = metrics["clip_score_whole"]

    if ref is None:
        _emit_ref_warning()
        metrics["psnr"] = float("nan")
        metrics["lpips"] = float("nan")
        return metrics

    if mask is None:
        raise ValueError("mask is required when ref is provided for PSNR/LPIPS.")

    metrics["psnr"] = psnr.compute(ref, pred, mask, masked_only=masked_only, device=device)
    metrics["lpips"] = lpips_metric.compute(
        ref, pred, mask, masked_only=masked_only, device=device
    )
    return metrics


def preload_models(device: str = "cpu") -> None:
    """Load heavy metric models before batched scoring."""
    clip_score.preload(device)
    # PSNR/LPIPS/FID backends load lazily when a reference image is present.


def unload_models() -> None:
    """Release cached metric models."""
    clip_score.unload()


def _clip_metrics_for_output(
    pred: Image.Image,
    prompt: str,
    *,
    mask: Image.Image | None,
    device: str,
) -> dict[str, float]:
    """Compute masked and whole-image CLIP scores for one generated output."""
    whole = clip_score.compute(
        pred,
        pred,
        mask=None,
        prompt=prompt,
        device=device,
        masked_only=False,
    )
    if mask is None:
        return {"clip_score_whole": whole, "clip_score": whole}
    return {
        "clip_score_whole": whole,
        "clip_score": clip_score.compute(
            pred,
            pred,
            mask=mask,
            prompt=prompt,
            device=device,
            masked_only=True,
        ),
    }


def rescore_geometric_clip_from_outputs(
    output_dir: Path,
    *,
    device: str = "cpu",
    manifest_path: Path | None = None,
    prompt_field: str = "caption_4v",
    result_fields: list[str] | None = None,
    sample_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Recompute ``clip_score`` and ``clip_score_whole`` from existing generated images.

    Does not run model inference; only reads images under ``output_dir/images/`` (or
    paths recorded in ``results.csv``).
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    results_csv = output_dir / "results.csv"

    rows: list[dict[str, Any]]
    if results_csv.is_file():
        rows = [dict(r) for r in read_csv_dicts(results_csv)]
    else:
        if manifest_path is None or not Path(manifest_path).is_file():
            raise FileNotFoundError(
                f"No {results_csv} and no manifest_path to join outputs under {images_dir}"
            )
        df = pd.read_csv(manifest_path, dtype={"sample_id": str})
        rows = []
        for _, sample in df.iterrows():
            sid = str(sample["sample_id"])
            out_path = images_dir / f"{sid}.png"
            if not out_path.is_file():
                continue
            prompt = str(sample.get(prompt_field) or sample.get("caption_4v") or "")
            rows.append(
                {
                    "sample_id": sid,
                    "prompt": prompt,
                    "caption_4v": str(sample.get("caption_4v", prompt)),
                    "tgt_mask_path": str(sample.get("tgt_mask_path", "")),
                    "output_path": str(out_path),
                }
            )

    if not rows:
        raise FileNotFoundError(f"No samples to rescore under {output_dir}")

    if sample_id is not None:
        rows = [r for r in rows if str(r.get("sample_id")) == str(sample_id)]
    elif limit is not None:
        rows = rows[: int(limit)]

    if not rows:
        raise FileNotFoundError(
            f"No samples to rescore after filters sample_id={sample_id!r} limit={limit!r}"
        )

    preload_models(device)
    try:
        for row in rows:
            out_path = Path(str(row.get("output_path") or ""))
            if not out_path.is_file():
                sid = str(row.get("sample_id", ""))
                out_path = images_dir / f"{sid}.png"
            if not out_path.is_file():
                continue

            prompt = str(row.get("prompt") or row.get("caption_4v") or "")
            ref_mask: Image.Image | None = None
            mask_path = row.get("tgt_mask_path")
            if mask_path and Path(str(mask_path)).is_file():
                with Image.open(mask_path) as m:
                    ref_mask = to_grayscale(m).copy()

            with Image.open(out_path) as out:
                pred = out.convert("RGB").copy()

            row.update(_clip_metrics_for_output(pred, prompt, mask=ref_mask, device=device))
    finally:
        unload_models()

    fields: list[str] = list(result_fields) if result_fields else []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    for name in ("clip_score", "clip_score_whole"):
        if name not in fields:
            fields.append(name)
    write_csv_dicts(results_csv, rows, fields)
    return rows


def _collect_finite_metric_values(
    results: list[dict[str, Any]],
    metric_name: str,
) -> list[float]:
    values: list[float] = []
    for row in results:
        if metric_name not in row:
            continue
        raw = row.get(metric_name)
        if raw is None:
            continue
        value = float(raw)
        if math.isnan(value):
            continue
        values.append(value)
    return values


def summarize_geometric(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return mean/std/min/max for each numeric metric, ignoring NaN."""
    summary: dict[str, dict[str, float]] = {}
    for name in GEOMETRIC_METRIC_NAMES:
        values = _collect_finite_metric_values(results, name)
        if not values:
            continue

        arr = np.array(values, dtype=np.float64)
        if name == "fid":
            summary[name] = {
                "value": float(arr[0]),
                "count": float(len(arr)),
            }
            continue

        summary[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": float(len(arr)),
        }
    return summary


def summarize_geometric_by_edit_type(
    results: list[dict[str, Any]],
    *,
    edit_param_by_sample: dict[str, str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return mean PSNR/LPIPS/CLIP grouped by edit_param transform type."""
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in EDIT_PARAM_TYPE_NAMES}
    for row in results:
        sample_id = str(row.get("sample_id", ""))
        edit_param = row.get("edit_param_json")
        if (not edit_param or edit_param == "") and edit_param_by_sample is not None:
            edit_param = edit_param_by_sample.get(sample_id, "")
        if not edit_param:
            continue
        edit_type = classify_edit_param_type(edit_param)
        grouped[edit_type].append(row)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for edit_type in EDIT_PARAM_TYPE_NAMES:
        rows = grouped[edit_type]
        if not rows:
            continue
        metrics_summary: dict[str, dict[str, float]] = {}
        for metric_name in EDIT_TYPE_METRIC_NAMES:
            values = _collect_finite_metric_values(rows, metric_name)
            if not values:
                continue
            arr = np.array(values, dtype=np.float64)
            metrics_summary[metric_name] = {
                "mean": float(np.mean(arr)),
                "count": float(len(arr)),
            }
        if metrics_summary:
            summary[edit_type] = metrics_summary
    return summary


def build_geometric_summary(
    results: list[dict[str, Any]],
    *,
    edit_param_by_sample: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build overall and per-edit-type geometric metric summaries."""
    summary: dict[str, Any] = summarize_geometric(results)
    by_edit_type = summarize_geometric_by_edit_type(
        results,
        edit_param_by_sample=edit_param_by_sample,
    )
    if by_edit_type:
        summary["by_edit_type"] = by_edit_type
    return summary
