"""Geometric editing evaluation runner for GeoBench manifests."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from eval.config import DEFAULT_DEVICE, DEFAULT_GEOBENCH_PROMPT_FIELD
from eval.logging_utils import get_logger
from eval.metrics.geometric_aggregator import (
    GEOMETRIC_METRIC_NAMES,
    build_geometric_summary,
    compute_geometric_per_sample,
    preload_models,
    rescore_geometric_clip_from_outputs,
    reset_ref_warning,
    unload_models,
)
from eval.metrics.geometric_verify import verify_geometric_output
from eval.methods.geometric_edit import get_geo_method
from eval.paths import ensure_dir, resolve_path
from eval.seed import set_global_seed
from eval.utils import append_csv_dict_row, read_csv_dicts, to_grayscale, write_csv_dicts

logger = get_logger(__name__)

RESULT_FIELDS = [
    "sample_id",
    "method",
    "prompt",
    "caption_4v",
    "ori_img_path",
    "ori_mask_path",
    "tgt_mask_path",
    "output_path",
    "psnr",
    "fid",
    "lpips",
    "clip_score",
    "clip_score_whole",
]


@dataclass
class GeneratedGeoSample:
    sample_id: str
    prompt: str
    caption_4v: str
    ori_img_path: str
    ori_mask_path: str
    tgt_mask_path: str
    output_path: str
    ref_img_path: str | None = None


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


def _edit_param_by_sample_from_manifest(manifest_path: Path) -> dict[str, str]:
    df = pd.read_csv(manifest_path, dtype={"sample_id": str})
    if "edit_param_json" not in df.columns:
        return {}
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        sample_id = str(row.get("sample_id", ""))
        edit_param = str(row.get("edit_param_json") or "")
        if sample_id and edit_param:
            mapping[sample_id] = edit_param
    return mapping


def _write_geometric_summaries(
    output_dir: Path,
    rows: list[dict],
    *,
    manifest_path: Path | None = None,
) -> Path:
    edit_param_by_sample = (
        _edit_param_by_sample_from_manifest(manifest_path)
        if manifest_path is not None and manifest_path.is_file()
        else None
    )
    summary = build_geometric_summary(rows, edit_param_by_sample=edit_param_by_sample)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def _resolve_row_prompt(row: pd.Series, prompt_field: str) -> str:
    if prompt_field == "edit_prompt":
        edit_prompt = str(row.get("edit_prompt", "") or "")
        if edit_prompt:
            return edit_prompt
    if "prompt" in row and str(row["prompt"] or ""):
        return str(row["prompt"])
    return str(row.get("caption_4v", "") or "")


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


def run_geometric_evaluation(
    *,
    method_name: str,
    manifest_path: Path,
    output_dir: Path,
    seed: int,
    limit: int | None,
    sample_id: str | None,
    device: str,
    dtype: torch.dtype,
    batch_name: str | None,
    prompt_field: str = DEFAULT_GEOBENCH_PROMPT_FIELD,
    resume: bool = True,
    offload: bool = False,
    debug_dir: str | None = None,
    verify: bool = False,
    rescore_clip_only: bool = False,
    summarize_only: bool = False,
) -> Path:
    """Run geometric editing evaluation and write results artifacts."""
    reset_ref_warning()
    set_global_seed(seed)
    output_dir = ensure_dir(output_dir)
    images_dir = ensure_dir(output_dir / "images")
    results_csv = output_dir / "results.csv"

    if summarize_only:
        if not results_csv.is_file():
            raise FileNotFoundError(f"No existing results to summarize: {results_csv}")
        rows = _load_existing_rows(results_csv)
        if not rows:
            raise ValueError(f"results.csv is empty: {results_csv}")
        summary_path = _write_geometric_summaries(
            output_dir,
            rows,
            manifest_path=manifest_path,
        )
        logger.info(
            "Summarize-only: wrote grouped metrics for %d samples to %s",
            len(rows),
            summary_path,
        )
        return results_csv

    if rescore_clip_only:
        logger.info(
            "Rescore-only: recomputing clip_score / clip_score_whole from existing images in %s",
            output_dir,
        )
        metric_device = device if device != "cpu" and torch.cuda.is_available() else "cpu"
        rows = rescore_geometric_clip_from_outputs(
            output_dir,
            device=metric_device,
            manifest_path=manifest_path,
            prompt_field=prompt_field,
            result_fields=RESULT_FIELDS,
            sample_id=sample_id,
            limit=limit,
        )
        summary_path = _write_geometric_summaries(
            output_dir,
            rows,
            manifest_path=manifest_path,
        )
        logger.info(
            "Rescored %d samples; wrote %s and %s",
            len(rows),
            results_csv,
            summary_path,
        )
        return results_csv

    df = pd.read_csv(manifest_path, dtype={"sample_id": str})
    if sample_id is not None:
        df = df[df["sample_id"] == str(sample_id)]
        if df.empty:
            raise ValueError(f"sample_id {sample_id} not found in {manifest_path}")
    elif limit is not None:
        df = df.head(limit)

    has_ref_col = "coarse_input_path" in df.columns and df["coarse_input_path"].notna().any()
    has_ref = has_ref_col or ("ref_img_path" in df.columns and df["ref_img_path"].notna().any())
    if not has_ref:
        logger.warning(
            "No reference edited image in manifest; skipping PSNR/LPIPS/FID. "
            "Computing CLIP only."
        )
    else:
        logger.info("Reference images available (coarse_input); PSNR/LPIPS enabled where applicable.")

    scored_ids = _load_scored_sample_ids(results_csv) if resume else set()
    all_samples: list[GeneratedGeoSample] = []
    pending_inference: list[pd.Series] = []
    skipped_outputs = 0

    for _, sample in df.iterrows():
        sample_id = str(sample["sample_id"])
        output_path = _sample_output_path(images_dir, sample_id)
        ref_path = None
        if has_ref_col and pd.notna(sample.get("coarse_input_path")):
            coarse = str(sample["coarse_input_path"])
            if coarse and Path(coarse).is_file():
                ref_path = coarse
        if ref_path is None and has_ref and pd.notna(sample.get("ref_img_path")):
            ref_path = str(sample["ref_img_path"])
        item = GeneratedGeoSample(
            sample_id=sample_id,
            prompt=_resolve_row_prompt(sample, prompt_field),
            caption_4v=str(sample.get("caption_4v", sample.get("prompt", ""))),
            ori_img_path=str(sample["ori_img_path"]),
            ori_mask_path=str(sample["ori_mask_path"]),
            tgt_mask_path=str(sample["tgt_mask_path"]),
            output_path=str(output_path),
            ref_img_path=ref_path,
        )

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

    method = get_geo_method(
        method_name,
        device=device,
        dtype=dtype,
        offload=offload,
    )

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
                prompt = _resolve_row_prompt(sample, prompt_field)
                output_path = _sample_output_path(images_dir, sample_id)

                with (
                    Image.open(sample["ori_img_path"]) as img,
                    Image.open(sample["ori_mask_path"]) as ori_msk,
                    Image.open(sample["tgt_mask_path"]) as tgt_msk,
                ):
                    ori_img = img.convert("RGB").copy()
                    ori_mask = to_grayscale(ori_msk).copy()
                    tgt_mask = to_grayscale(tgt_msk).copy()

                infer_kwargs: dict = {}
                if debug_dir:
                    infer_kwargs["debug_dir"] = str(
                        ensure_dir(Path(debug_dir) / sample_id)
                    )
                infer_kwargs["sample_id"] = sample_id
                infer_kwargs["cache_dir"] = ensure_dir(output_dir / "inp_backgrounds")
                if pd.notna(sample.get("edit_param_json")):
                    infer_kwargs["edit_param_json"] = sample["edit_param_json"]
                if pd.notna(sample.get("obj_label")):
                    infer_kwargs["obj_label"] = str(sample["obj_label"])

                result = method.infer(ori_img, ori_mask, tgt_mask, prompt, **infer_kwargs)
                result.save(output_path)

                ref_path = None
                if has_ref_col and pd.notna(sample.get("coarse_input_path")):
                    coarse = str(sample["coarse_input_path"])
                    if coarse and Path(coarse).is_file():
                        ref_path = coarse
                if ref_path is None and has_ref and pd.notna(sample.get("ref_img_path")):
                    ref_path = str(sample["ref_img_path"])
                all_samples.append(
                    GeneratedGeoSample(
                        sample_id=sample_id,
                        prompt=prompt,
                        caption_4v=str(sample.get("caption_4v", prompt)),
                        ori_img_path=str(sample["ori_img_path"]),
                        ori_mask_path=str(sample["ori_mask_path"]),
                        tgt_mask_path=str(sample["tgt_mask_path"]),
                        output_path=str(output_path),
                        ref_img_path=ref_path,
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

    pending_metrics = [
        item for item in all_samples if not resume or item.sample_id not in scored_ids
    ]
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
    new_rows: list[dict] = []

    if pending_metrics:
        logger.info("Loading evaluation metrics on %s", metric_device)
        preload_models(metric_device)

    try:
        for item in all_samples:
            with Image.open(item.output_path) as out:
                result = out.convert("RGB").copy()

            ref_image: Image.Image | None = None
            ref_mask: Image.Image | None = None
            if item.ref_img_path:
                with (
                    Image.open(item.ref_img_path) as ref_img,
                    Image.open(item.tgt_mask_path) as tgt_msk,
                ):
                    ref_image = ref_img.convert("RGB").copy()
                    ref_mask = to_grayscale(tgt_msk).copy()

            if item.sample_id in scored_ids:
                continue

            metrics = compute_geometric_per_sample(
                result,
                item.prompt,
                ref=ref_image,
                mask=ref_mask,
                device=metric_device,
            )
            if not has_ref:
                metrics["fid"] = float("nan")
            else:
                metrics["fid"] = float("nan")  # batch FID not wired for geo yet

            row = {
                "sample_id": item.sample_id,
                "method": method_name,
                "prompt": item.prompt,
                "caption_4v": item.caption_4v,
                "ori_img_path": item.ori_img_path,
                "ori_mask_path": item.ori_mask_path,
                "tgt_mask_path": item.tgt_mask_path,
                "output_path": item.output_path,
                **metrics,
            }
            new_rows.append(row)
            psnr_val = metrics["psnr"]
            psnr_str = f"{psnr_val:.3f}" if not math.isnan(psnr_val) else "nan"
            lpips_val = metrics["lpips"]
            lpips_str = f"{lpips_val:.4f}" if not math.isnan(lpips_val) else "nan"
            logger.info(
                "sample=%s psnr=%s lpips=%s clip=%.4f clip_whole=%.4f",
                item.sample_id,
                psnr_str,
                lpips_str,
                metrics["clip_score"],
                metrics["clip_score_whole"],
            )
            if resume:
                append_csv_dict_row(results_csv, row, RESULT_FIELDS)
    finally:
        if pending_metrics:
            unload_models()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics_sec = time.perf_counter() - metrics_start if pending_metrics else 0.0
    total_sec = time.perf_counter() - total_start

    existing_rows = _load_existing_rows(results_csv) if resume else []
    rows = _merge_result_rows(existing_rows, new_rows) if resume else new_rows

    timing = _build_timing_stats(
        num_samples=len(rows),
        inference_sec=inference_sec,
        metrics_sec=metrics_sec,
        total_sec=total_sec,
        model_load_sec=model_load_sec,
        inference_count=newly_generated,
        metrics_count=len(new_rows),
    )

    logger.info(
        "Phase 2/2 complete: scored %d new images in %.1fs (%.1fs/sample)",
        len(new_rows),
        metrics_sec,
        metrics_sec / len(new_rows) if new_rows else 0.0,
    )

    write_csv_dicts(results_csv, rows, RESULT_FIELDS)

    summary_path = _write_geometric_summaries(
        output_dir,
        rows,
        manifest_path=manifest_path,
    )

    config = {
        "method": method_name,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "limit": limit,
        "sample_id": sample_id,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "prompt_field": prompt_field,
        "offload": offload,
        "debug_dir": debug_dir,
        "batch_name": batch_name or output_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metric_names": list(GEOMETRIC_METRIC_NAMES),
        "has_reference_image": bool(has_ref),
        "num_samples": len(rows),
        "resume": resume,
        "timing": timing,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Wrote %s", results_csv)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", config_path)

    if verify and all_samples:
        verify_dir = ensure_dir(output_dir / "verify")
        for item in all_samples:
            verify_geometric_output(
                manifest_path=manifest_path,
                output_path=Path(item.output_path),
                sample_id=item.sample_id,
                out_dir=verify_dir,
            )

    return results_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run geometric editing evaluation for one method on GeoBench"
    )
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--batch-name", type=str, default=None)
    parser.add_argument("--offload", action="store_true", help="Enable model CPU offload (FLUX methods)")
    parser.add_argument("--debug-dir", type=str, default=None, help="Per-sample EditPipeline debug output root")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Write verification panels/metrics after inference",
    )
    parser.add_argument(
        "--prompt-field",
        type=str,
        default=DEFAULT_GEOBENCH_PROMPT_FIELD,
        choices=["caption_4v", "edit_prompt"],
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; regenerate all outputs and overwrite results.csv",
    )
    parser.add_argument(
        "--rescore-clip",
        action="store_true",
        help=(
            "Skip inference; recompute clip_score (tgt_mask bbox) and clip_score_whole "
            "(full image) from existing outputs under --output-dir"
        ),
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help=(
            "Skip inference and scoring; recompute summary.json from existing results.csv, "
            "including per edit_param-type PSNR/LPIPS/CLIP means"
        ),
    )
    args = parser.parse_args()

    run_geometric_evaluation(
        method_name=args.method,
        manifest_path=resolve_path(args.manifest),
        output_dir=resolve_path(args.output_dir),
        seed=args.seed,
        limit=args.limit,
        sample_id=args.sample_id,
        device=args.device,
        dtype=_resolve_dtype(args.dtype),
        batch_name=args.batch_name,
        prompt_field=args.prompt_field,
        resume=not args.no_resume,
        offload=args.offload,
        debug_dir=args.debug_dir,
        verify=args.verify,
        rescore_clip_only=args.rescore_clip,
        summarize_only=args.summarize_only,
    )


if __name__ == "__main__":
    main()
