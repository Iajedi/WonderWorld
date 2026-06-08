"""Single-method evaluation runner."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from eval.config import DEFAULT_DEVICE, DEFAULT_MASKED_METRICS
from eval.logging_utils import get_logger
from eval.metrics.aggregator import (
    METRIC_NAMES,
    compute_per_sample,
    preload_models,
    summarize,
    unload_models,
)
from eval.metrics.fid_metric import FIDAccumulator
from eval.methods import get_method
from eval.paths import ensure_dir, resolve_path
from eval.seed import set_global_seed
from eval.utils import append_csv_dict_row, read_csv_dicts, to_grayscale, write_csv_dicts

logger = get_logger(__name__)

RESULT_FIELDS = [
    "sample_id",
    "method",
    "prompt",
    "prompt_src",
    "prompt_tgt",
    "prompt_variant",
    "image_path",
    "mask_path",
    "output_path",
    "psnr",
    "fid",
    "lpips",
    "clip_score",
]


@dataclass
class GeneratedSample:
    sample_id: str
    prompt: str
    image_path: str
    mask_path: str
    output_path: str
    infer_image_path: str = ""
    prompt_src: str = ""
    prompt_tgt: str = ""
    prompt_variant: str = ""


def _prompt_fields_from_sample(sample: pd.Series) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in ("prompt_src", "prompt_tgt", "prompt_variant"):
        if key in sample and pd.notna(sample[key]):
            fields[key] = str(sample[key])
    return fields


def _infer_image_path(sample: pd.Series) -> str:
    """Return composed input path when present, otherwise the clean reference image."""
    composed = sample.get("composed_path")
    if composed is not None and pd.notna(composed) and str(composed).strip():
        return str(composed)
    return str(sample["image_path"])


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def _sample_output_path(images_dir: Path, sample_id: str) -> Path:
    return images_dir / f"{sample_id}.png"


def _output_exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _load_scored_sample_ids(results_csv: Path) -> set[str]:
    if not results_csv.is_file():
        return set()
    rows = read_csv_dicts(results_csv)
    return {str(row["sample_id"]) for row in rows if row.get("sample_id")}


def _load_existing_rows(results_csv: Path) -> list[dict]:
    if not results_csv.is_file():
        return []
    return read_csv_dicts(results_csv)


def _merge_result_rows(existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    """Merge rows by sample_id, preferring newly computed rows."""
    merged: dict[str, dict] = {}
    for row in existing_rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            merged[sample_id] = row
    for row in new_rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            merged[sample_id] = row
    return [merged[sid] for sid in sorted(merged)]


def _build_timing_stats(
    *,
    num_samples: int,
    inference_sec: float,
    metrics_sec: float,
    total_sec: float,
    model_load_sec: float = 0.0,
    inference_count: int | None = None,
    metrics_count: int | None = None,
) -> dict[str, float]:
    infer_n = inference_count if inference_count is not None else num_samples
    metric_n = metrics_count if metrics_count is not None else num_samples
    inference_per_sample = inference_sec / infer_n if infer_n else 0.0
    metrics_per_sample = metrics_sec / metric_n if metric_n else 0.0
    estimated_inference = inference_per_sample * 1000.0
    estimated_metrics = metrics_per_sample * 1000.0
    estimated_total = estimated_inference + estimated_metrics + model_load_sec
    return {
        "num_samples": float(num_samples),
        "inference_sec": inference_sec,
        "metrics_sec": metrics_sec,
        "total_sec": total_sec,
        "model_load_sec": model_load_sec,
        "inference_sec_per_sample": inference_per_sample,
        "metrics_sec_per_sample": metrics_per_sample,
        "total_sec_per_sample": total_sec / num_samples if num_samples else 0.0,
        "estimated_1000_inference_sec": estimated_inference,
        "estimated_1000_metrics_sec": estimated_metrics,
        "estimated_1000_total_sec": estimated_total,
        "estimated_1000_inference_hours": estimated_inference / 3600.0,
        "estimated_1000_metrics_hours": estimated_metrics / 3600.0,
        "estimated_1000_total_hours": estimated_total / 3600.0,
    }


def run_evaluation(
    *,
    method_name: str,
    manifest_path: Path,
    output_dir: Path,
    seed: int,
    limit: int | None,
    device: str,
    dtype: torch.dtype,
    masked_metrics: bool,
    batch_name: str | None,
    offload: bool = False,
    resume: bool = True,
    skip_metrics: bool = False,
    metrics_only: bool = False,
    rescore_metrics: bool = False,
) -> Path:
    """Run evaluation and write results artifacts."""
    set_global_seed(seed)
    output_dir = ensure_dir(output_dir)
    images_dir = ensure_dir(output_dir / "images")
    results_csv = output_dir / "results.csv"

    df = pd.read_csv(manifest_path, dtype={"sample_id": str, "mask_id": str})
    if limit is not None:
        df = df.head(limit)

    scored_ids = (
        set()
        if rescore_metrics
        else (_load_scored_sample_ids(results_csv) if resume else set())
    )
    all_samples: list[GeneratedSample] = []
    pending_inference: list[pd.Series] = []
    skipped_outputs = 0

    for _, sample in df.iterrows():
        sample_id = str(sample["sample_id"])
        output_path = _sample_output_path(images_dir, sample_id)
        item = GeneratedSample(
            sample_id=sample_id,
            prompt=str(sample["prompt"]),
            image_path=str(sample["image_path"]),
            mask_path=str(sample["mask_path"]),
            output_path=str(output_path),
            infer_image_path=_infer_image_path(sample),
            **_prompt_fields_from_sample(sample),
        )

        if metrics_only:
            if not _output_exists(output_path):
                raise FileNotFoundError(
                    f"--metrics-only: missing output image for sample {sample_id}: {output_path}"
                )
            all_samples.append(item)
            continue

        if resume and _output_exists(output_path):
            all_samples.append(item)
            skipped_outputs += 1
            continue

        pending_inference.append(sample)

    if resume and (skipped_outputs or scored_ids):
        logger.info(
            "Resume enabled: %d existing outputs, %d scored rows in %s",
            skipped_outputs,
            len(scored_ids),
            results_csv,
        )

    method = get_method(method_name, device=device, dtype=dtype)
    if offload and hasattr(method, "offload"):
        method.offload = True

    total_start = time.perf_counter()
    model_load_sec = 0.0
    inference_sec = 0.0
    newly_generated = 0

    if pending_inference:
        logger.info(
            "Phase 1/2: loading method '%s' on %s (%d samples to generate, %d skipped)",
            method_name,
            device,
            len(pending_inference),
            len(all_samples),
        )
        model_load_start = time.perf_counter()
        method.load()
        model_load_sec = time.perf_counter() - model_load_start
        inference_start = time.perf_counter()

        try:
            for sample in pending_inference:
                sample_id = str(sample["sample_id"])
                ref_image_path = Path(str(sample["image_path"]))
                infer_image_path = Path(_infer_image_path(sample))
                mask_path = Path(str(sample["mask_path"]))
                prompt = str(sample["prompt"])
                output_path = _sample_output_path(images_dir, sample_id)

                with Image.open(infer_image_path) as img, Image.open(mask_path) as msk:
                    image = img.convert("RGB").copy()
                    mask = to_grayscale(msk).copy()

                infer_kwargs: dict = {}
                if "prompt_src" in sample and pd.notna(sample["prompt_src"]):
                    infer_kwargs["prompt_src"] = str(sample["prompt_src"])
                if "prompt_tgt" in sample and pd.notna(sample["prompt_tgt"]):
                    infer_kwargs["prompt_tgt"] = str(sample["prompt_tgt"])
                result = method.infer(image, mask, prompt, **infer_kwargs)
                result.save(output_path)

                all_samples.append(
                    GeneratedSample(
                        sample_id=sample_id,
                        prompt=prompt,
                        image_path=str(ref_image_path),
                        mask_path=str(mask_path),
                        output_path=str(output_path),
                        infer_image_path=str(infer_image_path),
                        **_prompt_fields_from_sample(sample),
                    )
                )
                newly_generated += 1
                logger.info("Generated sample=%s -> %s", sample_id, output_path)
        finally:
            method.unload()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        inference_sec = time.perf_counter() - inference_start
    else:
        logger.info(
            "Phase 1/2: skipped inference (%d existing outputs)",
            len(all_samples),
        )

    logger.info(
        "Phase 1/2 complete: generated %d new images in %.1fs (%.1fs/sample)",
        newly_generated,
        inference_sec,
        inference_sec / newly_generated if newly_generated else 0.0,
    )

    metrics_sec = 0.0
    new_rows: list[dict] = []
    rows: list[dict] = []

    if skip_metrics:
        logger.info("Phase 2/2: skipped (--skip-metrics)")
    else:
        pending_metrics = (
            list(all_samples)
            if rescore_metrics
            else [
                item for item in all_samples if not resume or item.sample_id not in scored_ids
            ]
        )
        if not pending_metrics and all_samples:
            logger.info("Phase 2/2: all %d samples already scored", len(all_samples))
        elif pending_metrics:
            logger.info(
                "Phase 2/2: scoring %d samples (%d already in results.csv)",
                len(pending_metrics),
                len(scored_ids),
            )

        metric_device = device if device != "cpu" and torch.cuda.is_available() else "cpu"
        metrics_start = time.perf_counter()

        if pending_metrics:
            logger.info("Loading evaluation metrics on %s", metric_device)
            preload_models(metric_device)

        fid_accum = FIDAccumulator()
        try:
            for item in all_samples:
                with Image.open(item.image_path) as img, Image.open(item.mask_path) as msk, Image.open(
                    item.output_path
                ) as out:
                    image = img.convert("RGB").copy()
                    mask = to_grayscale(msk).copy()
                    result = out.convert("RGB").copy()

                fid_accum.add(image, result, mask, masked_only=masked_metrics)

                if item.sample_id in scored_ids:
                    continue

                metrics = compute_per_sample(
                    image,
                    result,
                    mask,
                    item.prompt,
                    masked_only=masked_metrics,
                    device=metric_device,
                )

                row = {
                    "sample_id": item.sample_id,
                    "method": method_name,
                    "prompt": item.prompt,
                    "prompt_src": item.prompt_src,
                    "prompt_tgt": item.prompt_tgt,
                    "prompt_variant": item.prompt_variant,
                    "image_path": item.image_path,
                    "mask_path": item.mask_path,
                    "output_path": item.output_path,
                    **metrics,
                }
                new_rows.append(row)
                logger.info(
                    "sample=%s psnr=%.3f lpips=%.4f clip=%.4f",
                    item.sample_id,
                    metrics["psnr"],
                    metrics["lpips"],
                    metrics["clip_score"],
                )
                if resume and not rescore_metrics:
                    append_csv_dict_row(results_csv, row, RESULT_FIELDS)
        finally:
            if pending_metrics:
                unload_models()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        metrics_sec = time.perf_counter() - metrics_start if pending_metrics else 0.0

        existing_rows = (
            []
            if rescore_metrics
            else (_load_existing_rows(results_csv) if resume else [])
        )
        rows = (
            new_rows
            if rescore_metrics or not resume
            else _merge_result_rows(existing_rows, new_rows)
        )

        if all_samples:
            batch_fid = fid_accum.compute(metric_device)
            logger.info("Batch FID=%.4f over %d samples", batch_fid, len(fid_accum))
            for row in rows:
                row["fid"] = batch_fid
        elif rows:
            batch_fid = float(rows[0].get("fid", 0.0))
        else:
            batch_fid = 0.0

        logger.info(
            "Phase 2/2 complete: scored %d new images in %.1fs (%.1fs/sample)",
            len(new_rows),
            metrics_sec,
            metrics_sec / len(new_rows) if new_rows else 0.0,
        )

        write_csv_dicts(results_csv, rows, RESULT_FIELDS)

        summary = summarize(rows)
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Wrote %s", results_csv)
        logger.info("Wrote %s", summary_path)

    total_sec = time.perf_counter() - total_start
    timing = _build_timing_stats(
        num_samples=len(all_samples),
        inference_sec=inference_sec,
        metrics_sec=metrics_sec,
        total_sec=total_sec,
        model_load_sec=model_load_sec,
        inference_count=newly_generated,
        metrics_count=len(new_rows),
    )

    logger.info(
        "Estimated 1000-image run: inference %.1f h, metrics %.1f h, total %.1f h",
        timing["estimated_1000_inference_hours"],
        timing["estimated_1000_metrics_hours"],
        timing["estimated_1000_total_hours"],
    )

    config = {
        "method": method_name,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "limit": limit,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "masked_metrics": masked_metrics,
        "batch_name": batch_name or output_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metric_names": list(METRIC_NAMES),
        "num_samples": len(all_samples),
        "resume": resume,
        "skip_metrics": skip_metrics,
        "metrics_only": metrics_only,
        "rescore_metrics": rescore_metrics,
        "timing": timing,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Wrote %s", config_path)
    return results_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inpainting evaluation for one method")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--batch-name", type=str, default=None)
    parser.add_argument("--offload", action="store_true", help="Enable model CPU offload (FLUX methods)")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; regenerate all outputs and overwrite results.csv",
    )
    parser.add_argument(
        "--masked-metrics",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MASKED_METRICS,
        help="Compute PSNR/LPIPS on masked region only; FID uses masked compositing",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Generate images only; skip PSNR/LPIPS/FID/CLIP scoring",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Score existing images only; do not load the inpainting method",
    )
    parser.add_argument(
        "--rescore-metrics",
        action="store_true",
        help="Recompute metrics for all samples, overwriting results.csv",
    )
    args = parser.parse_args()

    run_evaluation(
        method_name=args.method,
        manifest_path=resolve_path(args.manifest),
        output_dir=resolve_path(args.output_dir),
        seed=args.seed,
        limit=args.limit,
        device=args.device,
        dtype=_resolve_dtype(args.dtype),
        masked_metrics=args.masked_metrics,
        batch_name=args.batch_name,
        offload=args.offload,
        resume=not args.no_resume,
        skip_metrics=args.skip_metrics,
        metrics_only=args.metrics_only,
        rescore_metrics=args.rescore_metrics,
    )


if __name__ == "__main__":
    main()
