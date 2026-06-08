"""GeoBench 2D subset creation from Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.config import (
    DEFAULT_GEOBENCH_CONFIG,
    DEFAULT_GEOBENCH_COUNT,
    DEFAULT_GEOBENCH_DATASET,
    DEFAULT_GEOBENCH_SPLIT,
    DEFAULT_SUBSET_SEED,
)
from eval.data.geobench_geometry import edit_param_dict
from eval.data.manifest import GEOBENCH_SUBSET_FIELDS, GeoBenchRecord
from eval.logging_utils import get_logger
from eval.paths import ensure_dir, resolve_path
from eval.seed import set_global_seed
from eval.utils import binarize_mask, to_grayscale, to_rgb, write_csv_dicts, write_jsonl

logger = get_logger(__name__)


def _selected_indices(total: int, count: int, seed: int) -> list[int]:
    if count > total:
        raise ValueError(
            f"Requested count {count} exceeds GeoBench 2D split size {total}. "
            "Reduce --count or use the full dataset."
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total)
    return sorted(int(i) for i in perm[:count])


def _dataset_length(dataset_name: str, config_name: str, split_name: str) -> int:
    from datasets import load_dataset_builder

    builder = load_dataset_builder(dataset_name, config_name)
    if split_name not in builder.info.splits:
        available = ", ".join(builder.info.splits.keys())
        raise ValueError(
            f"Split '{split_name}' not found in {dataset_name} config '{config_name}'. "
            f"Available splits: {available}"
        )
    return int(builder.info.splits[split_name].num_examples)


def _load_hf_dataset(dataset_name: str, config_name: str, split_name: str):
    from datasets import load_dataset

    if config_name not in ("2d",):
        raise ValueError(
            f"Unsupported GeoBench config '{config_name}'. "
            "This loader only supports config '2d'."
        )
    return load_dataset(dataset_name, config_name, split=split_name)


def _pil_from_hf(value: Any) -> Image.Image:
    """Convert a Hugging Face image column value to PIL."""
    if value is None:
        raise ValueError("Missing image value in dataset row.")
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return Image.open(BytesIO(value["bytes"]))
        if "path" in value and value["path"]:
            return Image.open(value["path"])
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value))
    if isinstance(value, str):
        return Image.open(value)
    raise TypeError(f"Unsupported image type: {type(value)!r}")


def _serialize_edit_param(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _save_mask(mask: Image.Image, path: Path, *, reference_size: tuple[int, int] | None = None) -> Image.Image:
    """Canonicalize mask: single-channel, white = active/object region."""
    gray = to_grayscale(mask)
    if reference_size is not None and gray.size != reference_size:
        gray = gray.resize(reference_size, Image.Resampling.NEAREST)
    canonical = binarize_mask(gray, auto_polarity=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical.save(path)
    return canonical


def _save_row(
    row: dict[str, Any],
    dataset_index: int,
    *,
    sample_id: str,
    images_dir: Path,
    ori_masks_dir: Path,
    tgt_masks_dir: Path,
    coarse_inputs_dir: Path,
    dataset_name: str,
    config_name: str,
    split_name: str,
) -> GeoBenchRecord:
    ori_img = to_rgb(_pil_from_hf(row.get("ori_img")))
    img_size = ori_img.size

    ori_mask = _pil_from_hf(row.get("ori_mask"))
    tgt_mask = _pil_from_hf(row.get("tgt_mask"))

    img_path = images_dir / f"{sample_id}.png"
    ori_mask_path = ori_masks_dir / f"{sample_id}.png"
    tgt_mask_path = tgt_masks_dir / f"{sample_id}.png"
    coarse_input_path = coarse_inputs_dir / f"{sample_id}.png"

    ori_img.save(img_path)
    saved_ori_mask = _save_mask(ori_mask, ori_mask_path, reference_size=img_size)
    saved_tgt_mask = _save_mask(tgt_mask, tgt_mask_path, reference_size=img_size)

    coarse_input_saved = ""
    coarse_raw = row.get("coarse_input")
    if coarse_raw is not None:
        coarse_img = to_rgb(_pil_from_hf(coarse_raw))
        if coarse_img.size != img_size:
            coarse_img = coarse_img.resize(img_size, Image.Resampling.LANCZOS)
        coarse_img.save(coarse_input_path)
        coarse_input_saved = str(coarse_input_path)

    caption_4v = str(row.get("4v_caption") or row.get("caption_4v") or "")
    edit_prompt = str(row.get("edit_prompt") or "")
    obj_label = str(row.get("obj_label") or "")
    edit_param_json = _serialize_edit_param(row.get("edit_param"))
    edit_fields = edit_param_dict(row.get("edit_param")) if edit_param_json else {}

    return GeoBenchRecord(
        sample_id=sample_id,
        dataset_name=dataset_name,
        config_name=config_name,
        split_name=split_name,
        dataset_index=dataset_index,
        ori_img_path=str(img_path),
        ori_mask_path=str(ori_mask_path),
        tgt_mask_path=str(tgt_mask_path),
        coarse_input_path=coarse_input_saved,
        caption_4v=caption_4v,
        edit_prompt=edit_prompt,
        edit_param_json=edit_param_json,
        edit_dx=float(edit_fields.get("edit_dx", 0.0)),
        edit_dy=float(edit_fields.get("edit_dy", 0.0)),
        edit_dz=float(edit_fields.get("edit_dz", 0.0)),
        edit_rx=float(edit_fields.get("edit_rx", 0.0)),
        edit_ry=float(edit_fields.get("edit_ry", 0.0)),
        edit_rz=float(edit_fields.get("edit_rz", 0.0)),
        edit_sx=float(edit_fields.get("edit_sx", 1.0)),
        edit_sy=float(edit_fields.get("edit_sy", 1.0)),
        edit_sz=float(edit_fields.get("edit_sz", 1.0)),
        obj_label=obj_label,
        original_width=ori_img.width,
        original_height=ori_img.height,
        mask_width=saved_ori_mask.width,
        mask_height=saved_ori_mask.height,
    )


def build_geobench_2d_subset(
    out_dir: Path,
    *,
    count: int = DEFAULT_GEOBENCH_COUNT,
    seed: int = DEFAULT_SUBSET_SEED,
    dataset_name: str = DEFAULT_GEOBENCH_DATASET,
    config_name: str = DEFAULT_GEOBENCH_CONFIG,
    split_name: str = DEFAULT_GEOBENCH_SPLIT,
    force: bool = False,
) -> Path:
    """Create a fixed GeoBench 2D subset with local images, masks, and metadata."""
    set_global_seed(seed)
    if force and out_dir.exists():
        logger.info("Removing existing output directory (--force): %s", out_dir)
        shutil.rmtree(out_dir)

    out_dir = ensure_dir(out_dir)
    images_dir = ensure_dir(out_dir / "images")
    ori_masks_dir = ensure_dir(out_dir / "ori_masks")
    tgt_masks_dir = ensure_dir(out_dir / "tgt_masks")
    coarse_inputs_dir = ensure_dir(out_dir / "coarse_inputs")

    total = _dataset_length(dataset_name, config_name, split_name)
    selected = _selected_indices(total, count, seed)
    logger.info(
        "dataset=%s config=%s split=%s total=%d selected=%d seed=%d",
        dataset_name,
        config_name,
        split_name,
        total,
        len(selected),
        seed,
    )

    try:
        ds = _load_hf_dataset(dataset_name, config_name, split_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {dataset_name} (config={config_name}, split={split_name}) "
            f"from Hugging Face: {exc}. "
            "Check network access, dataset availability, and `datasets` installation."
        ) from exc

    records: list[dict[str, Any]] = []
    for out_idx, ds_idx in enumerate(selected):
        row = ds[int(ds_idx)]
        record = _save_row(
            row,
            int(ds_idx),
            sample_id=f"{out_idx:06d}",
            images_dir=images_dir,
            ori_masks_dir=ori_masks_dir,
            tgt_masks_dir=tgt_masks_dir,
            coarse_inputs_dir=coarse_inputs_dir,
            dataset_name=dataset_name,
            config_name=config_name,
            split_name=split_name,
        )
        records.append(record.to_dict())

    if len(records) != count:
        raise RuntimeError(
            f"Expected {count} records but saved {len(records)}. "
            "Check dataset split/config or Hugging Face cache integrity."
        )

    if count >= 1000 and count % 1000 == 0:
        stem = f"geobench_{config_name}_{count // 1000}k"
    else:
        stem = f"geobench_{config_name}_{count}"
    jsonl_path = out_dir / f"{stem}.jsonl"
    csv_path = out_dir / f"{stem}.csv"
    write_jsonl(jsonl_path, records)
    write_csv_dicts(csv_path, records, GEOBENCH_SUBSET_FIELDS)
    logger.info("Wrote %d records to %s and %s", len(records), jsonl_path, csv_path)
    return jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create GeoBench 2D subset from Hugging Face (config 2d only)"
    )
    parser.add_argument("--out", type=str, default="eval/output/geobench_2d_1k")
    parser.add_argument("--count", type=int, default=DEFAULT_GEOBENCH_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--dataset", type=str, default=DEFAULT_GEOBENCH_DATASET)
    parser.add_argument("--config", type=str, default=DEFAULT_GEOBENCH_CONFIG)
    parser.add_argument("--split", type=str, default=DEFAULT_GEOBENCH_SPLIT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output directory before re-downloading",
    )
    args = parser.parse_args()

    if args.config != "2d":
        raise SystemExit(
            f"Only GeoBench config '2d' is supported (got '{args.config}'). "
            "Use --config 2d."
        )

    build_geobench_2d_subset(
        resolve_path(args.out),
        count=args.count,
        seed=args.seed,
        dataset_name=args.dataset,
        config_name=args.config,
        split_name=args.split,
        force=args.force,
    )


if __name__ == "__main__":
    main()
