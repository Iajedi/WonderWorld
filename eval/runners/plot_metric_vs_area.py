"""Plot inpainting metrics against mask area from existing eval results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from eval.config import DEFAULT_MASKED_METRICS
from eval.logging_utils import get_logger
from eval.metrics.aggregator import METRIC_NAMES
from eval.metrics.fid_metric import FIDAccumulator, preload as preload_fid, unload as unload_fid
from eval.paths import resolve_path
from eval.utils import mask_area_ratio, to_grayscale

logger = get_logger(__name__)

PER_SAMPLE_METRICS = ("psnr", "lpips", "clip_score")
DISPLAY_NAMES = {
    "psnr": "PSNR",
    "fid": "FID",
    "lpips": "LPIPS",
    "clip_score": "CLIP score",
}


def _load_run_config(results_dir: Path) -> dict:
    config_path = results_dir / "config.json"
    if not config_path.is_file():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _attach_mask_area(
    results: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    benchmark: str,
) -> pd.DataFrame:
    """Join results with per-sample mask area (fraction in [0, 1])."""
    manifest = manifest.copy()
    results = results.copy()
    results["sample_id"] = results["sample_id"].astype(str)
    manifest["sample_id"] = manifest["sample_id"].astype(str)

    if benchmark == "ablation_outpaint":
        if "outpaint_coverage" not in manifest.columns:
            raise ValueError("ablation_outpaint manifest must include outpaint_coverage.")
        merged = results.merge(
            manifest[["sample_id", "outpaint_coverage"]],
            on="sample_id",
            how="left",
        )
        merged["mask_area"] = merged["outpaint_coverage"].astype(float)
        if merged["mask_area"].isna().any():
            raise ValueError("Failed to join outpaint_coverage for all result rows.")
        return merged

    if benchmark == "flickr30k":
        mask_paths = results["mask_path"] if "mask_path" in results.columns else None
        if mask_paths is None:
            merged = results.merge(manifest[["sample_id", "mask_path"]], on="sample_id", how="left")
            mask_paths = merged["mask_path"]
        else:
            merged = results.copy()
        areas: list[float] = []
        for mask_path in mask_paths:
            with Image.open(mask_path) as msk:
                areas.append(mask_area_ratio(to_grayscale(msk)))
        merged["mask_area"] = areas
        return merged

    raise ValueError(f"Unknown benchmark {benchmark!r}. Use ablation_outpaint or flickr30k.")


def _assign_area_bins(df: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    """Assign each sample to one of ``num_bins`` equal-width mask-area bins."""
    out = df.copy()
    _, bin_edges = np.histogram(out["mask_area"], bins=num_bins)
    # Ensure the top edge includes the maximum observed area.
    bin_edges[-1] = max(bin_edges[-1], float(out["mask_area"].max()) + 1e-9)
    cats = pd.cut(out["mask_area"], bins=bin_edges, include_lowest=True)
    out["area_bin"] = cats.cat.codes
    out["mask_area"] = cats.apply(lambda interval: interval.mid).astype(float)
    out["bin_label"] = cats.apply(lambda interval: f"({math.floor(interval.left * 100)/100.0}, {math.floor(interval.right * 100)/100.0}]").astype(str)
    return out


def _mean_per_group(df: pd.DataFrame, metric: str, *, group_col: str) -> pd.DataFrame:
    agg: dict[str, tuple[str, str]] = {
        "mean": (metric, "mean"),
        "std": (metric, "std"),
        "count": (metric, "count"),
    }
    if group_col != "mask_area":
        agg["mask_area"] = ("mask_area", "mean")
    grouped = df.groupby(group_col, observed=True).agg(**agg).reset_index()
    if group_col == "mask_area":
        grouped["mask_area"] = grouped[group_col].astype(float)
    if group_col == "area_bin":
        labels = (
            df.groupby(group_col, observed=True)["bin_label"]
            .first()
            .reset_index()
        )
        grouped = grouped.merge(labels, on=group_col, how="left")
    return grouped.sort_values(group_col)


def _recompute_fid_per_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    masked_only: bool,
    device: str,
) -> pd.DataFrame:
    """Compute batch FID separately for each mask-area group."""
    preload_fid(device)
    rows: list[dict] = []
    try:
        for group_key, group in df.groupby(group_col, observed=True):
            accum = FIDAccumulator()
            for _, sample in group.iterrows():
                with Image.open(sample["image_path"]) as img, Image.open(
                    sample["mask_path"]
                ) as msk, Image.open(sample["output_path"]) as out:
                    ref = img.convert("RGB").copy()
                    mask = to_grayscale(msk).copy()
                    pred = out.convert("RGB").copy()
                accum.add(ref, pred, mask, masked_only=masked_only)

            fid = accum.compute(device)
            row = {
                group_col: group_key,
                "mask_area": float(group["mask_area"].mean()),
                "mean": fid,
                "std": 0.0,
                "count": len(group),
            }
            if "bin_label" in group.columns:
                row["bin_label"] = str(group["bin_label"].iloc[0])
            rows.append(row)
            logger.info(
                "FID %s=%s n=%d -> %.4f",
                group_col,
                group_key,
                len(group),
                fid,
            )
    finally:
        unload_fid()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows).sort_values(group_col)


def _plot_metric_line(
    stats: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    x_pct = stats["mask_area"].to_numpy() * 100.0
    y = stats["mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_pct, y, marker="o", linewidth=2)
    ax.set_xlabel("Mask area (%)")
    ax.set_ylabel(DISPLAY_NAMES[metric])
    ax.set_title(f"{DISPLAY_NAMES[metric]} against mask area")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_metric_histogram(
    stats: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    if "bin_label" in stats.columns:
        x_labels = stats["bin_label"].tolist()
    else:
        x_labels = [f"{v * 100:.1f}" for v in stats["mask_area"].tolist()]
    y = stats["mean"].to_numpy()
    x_pos = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos, y, width=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Mask area bin")
    ax.set_ylabel(DISPLAY_NAMES[metric])
    ax.set_title(f"{DISPLAY_NAMES[metric]} against mask area")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_metrics_vs_area(
    *,
    results_dir: Path,
    manifest_path: Path,
    benchmark: str,
    plot_dir: Path | None = None,
    device: str = "cpu",
    masked_metrics: bool | None = None,
    num_bins: int | None = None,
    plot_style: str = "line",
) -> Path:
    """Build per-metric plots and a summary CSV grouped by mask area."""
    results_dir = resolve_path(results_dir)
    manifest_path = resolve_path(manifest_path)
    plot_dir = resolve_path(plot_dir or results_dir / "plots")

    results_csv = results_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Missing results.csv in {results_dir}")

    run_cfg = _load_run_config(results_dir)
    if masked_metrics is None:
        masked_metrics = bool(run_cfg.get("masked_metrics", DEFAULT_MASKED_METRICS))

    results = pd.read_csv(results_csv, dtype={"sample_id": str, "mask_id": str})
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    df = _attach_mask_area(results, manifest, benchmark=benchmark)

    if num_bins is not None and num_bins > 0:
        df = _assign_area_bins(df, num_bins)
        group_col = "area_bin"
    else:
        group_col = "mask_area"

    plot_fn = _plot_metric_histogram if plot_style == "histogram" else _plot_metric_line

    summary_rows: list[dict] = []
    for metric in METRIC_NAMES:
        if metric == "fid":
            stats = _recompute_fid_per_group(
                df,
                group_col=group_col,
                masked_only=masked_metrics,
                device=device,
            )
        else:
            stats = _mean_per_group(df, metric, group_col=group_col)

        for _, row in stats.iterrows():
            summary_rows.append(
                {
                    "metric": metric,
                    "group": row[group_col],
                    "mask_area": row["mask_area"],
                    "mask_area_pct": row["mask_area"] * 100.0,
                    "bin_label": row.get("bin_label", ""),
                    "mean": row["mean"],
                    "std": row["std"],
                    "count": row["count"],
                }
            )

        plot_fn(stats, metric, plot_dir / f"{metric}_vs_mask_area.svg")

    summary_path = plot_dir / "metric_vs_area_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    logger.info("Wrote %s", summary_path)
    return plot_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot inpainting metrics against mask area from existing results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Eval output directory containing results.csv (and optional config.json)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Benchmark manifest CSV (e.g. ablation_outpaint.csv or benchmark_1k.csv)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["ablation_outpaint", "flickr30k"],
        default="ablation_outpaint",
        help="How to obtain per-sample mask area",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory for PNG plots (default: <results-dir>/plots)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--masked-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use masked compositing for FID (default: read from results config.json)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Aggregate mask areas into N equal-width bins (e.g. 10 for Flickr30k histograms)",
    )
    parser.add_argument(
        "--plot-style",
        type=str,
        choices=["line", "histogram"],
        default=None,
        help="Plot style (default: histogram when --bins is set, else line)",
    )
    args = parser.parse_args()

    plot_style = args.plot_style or ("histogram" if args.bins else "line")

    plot_metrics_vs_area(
        results_dir=resolve_path(args.results_dir),
        manifest_path=resolve_path(args.manifest),
        benchmark=args.benchmark,
        plot_dir=resolve_path(args.plot_dir) if args.plot_dir else None,
        device=args.device,
        masked_metrics=args.masked_metrics,
        num_bins=args.bins,
        plot_style=plot_style,
    )


if __name__ == "__main__":
    main()
