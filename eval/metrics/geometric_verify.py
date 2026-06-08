"""Visual verification for GeoBench geometric editing outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eval.logging_utils import get_logger
from eval.paths import ensure_dir, resolve_path
from eval.utils import binarize_mask, read_csv_dicts, to_grayscale, to_rgb

logger = get_logger(__name__)


def _load_manifest_row(manifest_path: Path, sample_id: str) -> dict:
    rows = read_csv_dicts(manifest_path)
    for row in rows:
        if str(row.get("sample_id")) == sample_id:
            return row
    raise ValueError(f"sample_id {sample_id} not found in {manifest_path}")


def _overlay_mask(image: Image.Image, mask: Image.Image, *, color: tuple[int, int, int] = (255, 64, 64)) -> Image.Image:
    rgb = to_rgb(image).copy()
    m = np.asarray(binarize_mask(to_grayscale(mask)), dtype=np.uint8) >= 128
    arr = np.array(rgb, dtype=np.float32)
    tint = np.array(color, dtype=np.float32)
    arr[m] = 0.55 * arr[m] + 0.45 * tint
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _make_panel(
    ori_img: Image.Image,
    output: Image.Image,
    coarse_input: Image.Image | None,
    tgt_mask: Image.Image,
) -> Image.Image:
    h, w = ori_img.size[1], ori_img.size[0]
    overlay = _overlay_mask(output, tgt_mask)
    panels: list[Image.Image] = [to_rgb(ori_img), to_rgb(output), overlay]
    labels = ["ori_img", "output", "output+tgt_mask"]

    if coarse_input is not None:
        panels.insert(2, to_rgb(coarse_input))
        labels.insert(2, "coarse_input")

    gap = 8
    label_h = 24
    total_w = len(panels) * w + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, h + label_h), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for panel, label in zip(panels, labels):
        if panel.size != (w, h):
            panel = panel.resize((w, h), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x, label_h))
        draw.text((x + 4, 4), label, fill=(255, 255, 255))
        x += w + gap

    return canvas


def _psnr(ref: Image.Image, pred: Image.Image) -> float:
    a = np.array(to_rgb(ref), dtype=np.float64)
    b = np.array(to_rgb(pred), dtype=np.float64)
    if a.shape != b.shape:
        b_img = Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.Resampling.LANCZOS)
        b = np.array(b_img, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def verify_geometric_output(
    *,
    manifest_path: Path,
    output_path: Path,
    sample_id: str,
    out_dir: Path,
) -> dict:
    """Build a comparison panel and alignment metrics for one sample."""
    row = _load_manifest_row(manifest_path, sample_id)
    out_dir = ensure_dir(out_dir)

    with (
        Image.open(row["ori_img_path"]) as ori,
        Image.open(output_path) as out_img,
        Image.open(row["tgt_mask_path"]) as tgt_m,
    ):
        ori_img = ori.convert("RGB").copy()
        output = out_img.convert("RGB").copy()
        tgt_mask = to_grayscale(tgt_m).copy()

    coarse_input: Image.Image | None = None
    coarse_path = str(row.get("coarse_input_path") or "")
    if coarse_path and Path(coarse_path).is_file():
        with Image.open(coarse_path) as coarse:
            coarse_input = coarse.convert("RGB").copy()

    panel = _make_panel(ori_img, output, coarse_input, tgt_mask)
    panel_path = out_dir / f"{sample_id}_panel.png"
    panel.save(panel_path)

    metrics: dict[str, float | str] = {
        "sample_id": sample_id,
        "output_path": str(output_path),
        "panel_path": str(panel_path),
    }

    ori_arr = np.array(ori_img, dtype=np.float32)
    out_arr = np.array(output, dtype=np.float32)
    diff = np.mean(np.abs(ori_arr - out_arr), axis=2)
    edited_region = diff > 12.0
    tgt_bool = np.asarray(binarize_mask(tgt_mask), dtype=bool)
    if edited_region.any() and tgt_bool.any():
        inter = np.logical_and(edited_region, tgt_bool).sum()
        union = np.logical_or(edited_region, tgt_bool).sum()
        metrics["edited_region_tgt_iou"] = float(inter / union) if union else 0.0
    else:
        metrics["edited_region_tgt_iou"] = 0.0

    if coarse_input is not None:
        metrics["psnr_vs_coarse_input"] = _psnr(coarse_input, output)
        metrics["coarse_input_path"] = coarse_path

    metrics["edit_param_iou"] = float(row.get("edit_param_iou") or 0.0)

    metrics_path = out_dir / f"{sample_id}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Wrote panel %s and metrics %s", panel_path, metrics_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GeoBench geometric edit output")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Generated output image path")
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="Verification output directory")
    args = parser.parse_args()

    metrics = verify_geometric_output(
        manifest_path=resolve_path(args.manifest),
        output_path=resolve_path(args.output),
        sample_id=args.sample_id,
        out_dir=resolve_path(args.out),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
