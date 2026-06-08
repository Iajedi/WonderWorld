"""Panoramic novel-view evaluation runner."""

from __future__ import annotations

import argparse
import json
import math
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from eval.config import DEFAULT_DEVICE
from eval.logging_utils import get_logger
from eval.metrics import aesthetic_score, clip_consistency, clip_iqa_plus, clip_score
from eval.paths import ensure_dir, resolve_path
from eval.seed import set_global_seed
from eval.utils import write_csv_dicts
from generate_pano import build_arg_parser as pano_arg_parser, generate_and_render_panorama

logger = get_logger(__name__)

RESULT_FIELDS = [
    "scene_id",
    "prompt",
    "view_index",
    "yaw_deg",
    "cs",
    "cc",
    "ciqa",
    "cas",
    "image_path",
]

METRIC_NAMES = ("CS", "CC", "CIQA", "CAS")


def _summarize_pano(rows: list[dict]) -> dict:
    """Compute mean/std for panoramic metrics."""
    key_map = {"CS": "cs", "CC": "cc", "CIQA": "ciqa", "CAS": "cas"}
    metrics: dict[str, dict[str, float]] = {}
    for display_name, field in key_map.items():
        values = []
        for row in rows:
            val = row.get(field)
            if val is None:
                continue
            fval = float(val)
            if math.isnan(fval):
                continue
            values.append(fval)
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        metrics[display_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    return metrics


def _load_views(scene_dir: Path) -> tuple[dict, list[Image.Image]]:
    manifest_path = scene_dir / "views_manifest.json"
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    views: list[Image.Image] = []
    for entry in manifest["views"]:
        img_path = scene_dir / entry["image_path"]
        views.append(Image.open(img_path).convert("RGB"))
    return manifest, views


def _score_scene(
    scene_id: str,
    prompt: str,
    scene_dir: Path,
    device: str,
    clip_l_model,
    clip_l_preprocess,
    aesthetic_mlp,
) -> list[dict]:
    manifest, views = _load_views(scene_dir)
    central_index = int(manifest["central_view_index"])

    cc_result = clip_consistency.compute_clip_consistency(
        views,
        central_index,
        clip_l_model,
        clip_l_preprocess,
        device,
    )
    ciqa_result = clip_iqa_plus.compute_clip_iqa_plus(views, device)
    cas_result = aesthetic_score.compute_aesthetic_scores(
        views,
        clip_l_model,
        clip_l_preprocess,
        aesthetic_mlp,
        device,
    )

    rows: list[dict] = []
    dummy_ref = views[central_index]
    for entry in manifest["views"]:
        view_idx = int(entry["view_index"])
        view_image = views[view_idx]
        cs = clip_score.compute(
            dummy_ref,
            view_image,
            mask=None,
            prompt=prompt,
            device=device,
        )
        cc = cc_result["per_view_cc"][view_idx]
        ciqa = ciqa_result["per_view_ciqa"][view_idx]
        cas = cas_result["per_view_cas"][view_idx]
        rows.append(
            {
                "scene_id": scene_id,
                "prompt": prompt,
                "view_index": view_idx,
                "yaw_deg": entry["yaw_deg"],
                "cs": cs,
                "cc": cc,
                "ciqa": ciqa,
                "cas": cas,
                "image_path": str(scene_dir / entry["image_path"]),
            }
        )
    return rows


def _preload_metric_models(device: str) -> tuple:
    clip_score.preload(device)
    clip_l_model, clip_l_preprocess = clip_consistency.preload(device)
    clip_iqa_plus.preload(device)
    aesthetic_mlp = aesthetic_score.preload(device)
    return clip_l_model, clip_l_preprocess, aesthetic_mlp


def _unload_metric_models() -> None:
    clip_score.unload()
    clip_consistency.unload()
    clip_iqa_plus.unload()
    aesthetic_score.unload()


def run_pano_evaluation(
    *,
    manifest: Path,
    output_dir: Path,
    batch_name: str,
    seed: int = 42,
    device: str = DEFAULT_DEVICE,
    limit: int | None = None,
    yaw_range: float = 60.0,
    num_views: int = 9,
    skip_generate: bool = False,
    base_config: str = "./config/base-config.yaml",
) -> None:
    set_global_seed(seed)
    ensure_dir(output_dir)

    df = pd.read_csv(manifest, dtype={"scene_id": str})
    if limit is not None:
        df = df.head(limit)

    results_csv = output_dir / f"results_{batch_name}.csv"
    summary_path = output_dir / f"summary_{batch_name}.json"
    config_path = output_dir / f"config_{batch_name}.json"

    total_start = time.perf_counter()
    gen_start = time.perf_counter()

    if not skip_generate:
        pano_parser = pano_arg_parser()
        for _, row in df.iterrows():
            scene_id = str(row["scene_id"])
            image_path = resolve_path(str(row["image_path"]))
            prompt = str(row["prompt"])
            logger.info("Generating panorama for scene %s", scene_id)
            gen_args = Namespace(
                input_image=str(image_path),
                prompt=prompt,
                output_dir=str(output_dir),
                scene_id=scene_id,
                yaw_range=yaw_range,
                num_views=num_views,
                seed=seed,
                device=device,
                base_config=base_config,
            )
            generate_and_render_panorama(gen_args)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gen_sec = time.perf_counter() - gen_start

    metrics_start = time.perf_counter()
    clip_l_model, clip_l_preprocess, aesthetic_mlp = _preload_metric_models(device)
    all_rows: list[dict] = []
    try:
        for _, row in df.iterrows():
            scene_id = str(row["scene_id"])
            prompt = str(row["prompt"])
            scene_dir = output_dir / scene_id
            if not (scene_dir / "views_manifest.json").is_file():
                logger.warning("Skipping scene %s — views_manifest.json missing.", scene_id)
                continue
            logger.info("Scoring scene %s", scene_id)
            scene_rows = _score_scene(
                scene_id,
                prompt,
                scene_dir,
                device,
                clip_l_model,
                clip_l_preprocess,
                aesthetic_mlp,
            )
            all_rows.extend(scene_rows)
    finally:
        _unload_metric_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_sec = time.perf_counter() - metrics_start
    total_sec = time.perf_counter() - total_start

    write_csv_dicts(results_csv, all_rows, RESULT_FIELDS)

    num_scenes = len({r["scene_id"] for r in all_rows})
    summary = {
        "batch_name": batch_name,
        "num_scenes": num_scenes,
        "num_views_per_scene": num_views,
        "metrics": _summarize_pano(all_rows),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    config = {
        "batch_name": batch_name,
        "manifest": str(manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": seed,
        "device": device,
        "limit": limit,
        "yaw_range": yaw_range,
        "num_views": num_views,
        "skip_generate": skip_generate,
        "base_config": base_config,
        "num_scenes_processed": num_scenes,
        "num_view_rows": len(all_rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timing": {
            "generation_sec": gen_sec,
            "metrics_sec": metrics_sec,
            "total_sec": total_sec,
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Wrote %s", results_csv)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Panoramic novel-view benchmark runner.")
    parser.add_argument(
        "--manifest",
        default="eval/manifests/processed_pano/manifest.csv",
        help="Path to manifest CSV.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for views and results.")
    parser.add_argument("--batch-name", required=True, help="Batch name for result files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--yaw-range", type=float, default=60.0)
    parser.add_argument("--num-views", type=int, default=9)
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generation; score existing views in output-dir.",
    )
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="WonderWorld base config passed to generate_pano.py.",
    )
    args = parser.parse_args()

    run_pano_evaluation(
        manifest=resolve_path(args.manifest),
        output_dir=Path(args.output_dir).resolve(),
        batch_name=args.batch_name,
        seed=args.seed,
        device=args.device,
        limit=args.limit,
        yaw_range=args.yaw_range,
        num_views=args.num_views,
        skip_generate=args.skip_generate,
        base_config=args.base_config,
    )


if __name__ == "__main__":
    main()
