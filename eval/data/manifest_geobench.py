"""Build the GeoBench geometric-editing benchmark manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from eval.config import DEFAULT_GEOBENCH_PROMPT_FIELD, DEFAULT_SUBSET_SEED, DEFAULT_TARGET_SIZE
from eval.data.geobench_geometry import mask_iou, predict_tgt_mask_from_edit_param
from eval.data.manifest import GEOBENCH_MANIFEST_FIELDS, GeoBenchBenchmarkSample
from eval.logging_utils import get_logger
from eval.masks.transforms import resize_center_crop, transform_image_and_dual_masks
from eval.paths import ensure_dir, resolve_path
from eval.seed import set_global_seed
from eval.utils import read_jsonl, to_rgb, write_csv_dicts

logger = get_logger(__name__)

IOU_WARN_THRESHOLD = 0.85


def _resolve_prompt(row: dict, prompt_field: str) -> str:
    if prompt_field == "edit_prompt":
        prompt = str(row.get("edit_prompt") or "")
        if prompt:
            return prompt
        return str(row.get("caption_4v") or "")
    return str(row.get("caption_4v") or "")


def build_geobench_manifest(
    input_jsonl: Path,
    out_csv: Path,
    *,
    seed: int = DEFAULT_SUBSET_SEED,
    size: int = DEFAULT_TARGET_SIZE,
    prompt_field: str = DEFAULT_GEOBENCH_PROMPT_FIELD,
    processed_dir: Path | None = None,
) -> Path:
    """Resize GeoBench subset assets and write the benchmark manifest CSV."""
    set_global_seed(seed)

    rows = read_jsonl(input_jsonl)
    if not rows:
        raise ValueError(f"No records found in {input_jsonl}")

    base_processed = ensure_dir(processed_dir or out_csv.parent / "processed")
    proc_images = ensure_dir(base_processed / "images")
    proc_ori_masks = ensure_dir(base_processed / "ori_masks")
    proc_tgt_masks = ensure_dir(base_processed / "tgt_masks")
    proc_coarse = ensure_dir(base_processed / "coarse_inputs")

    benchmark_rows: list[dict] = []
    iou_values: list[float] = []
    low_iou_count = 0

    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"Row missing sample_id in {input_jsonl}")

        with (
            Image.open(row["ori_img_path"]) as img,
            Image.open(row["ori_mask_path"]) as ori_msk,
            Image.open(row["tgt_mask_path"]) as tgt_msk,
        ):
            out_image, out_ori_mask, out_tgt_mask, _ = transform_image_and_dual_masks(
                img, ori_msk, tgt_msk, size, size
            )

        edit_param_iou = 0.0
        edit_param_json = str(row.get("edit_param_json") or "")
        if edit_param_json:
            try:
                predicted = predict_tgt_mask_from_edit_param(
                    out_ori_mask, edit_param_json, size=size
                )
                edit_param_iou = mask_iou(predicted, out_tgt_mask)
                iou_values.append(edit_param_iou)
                if edit_param_iou < IOU_WARN_THRESHOLD:
                    low_iou_count += 1
                    logger.warning(
                        "sample=%s edit_param IoU=%.3f below %.2f",
                        sample_id,
                        edit_param_iou,
                        IOU_WARN_THRESHOLD,
                    )
            except Exception as exc:
                logger.warning("sample=%s edit_param validation failed: %s", sample_id, exc)

        image_out = proc_images / f"{sample_id}.png"
        ori_mask_out = proc_ori_masks / f"{sample_id}.png"
        tgt_mask_out = proc_tgt_masks / f"{sample_id}.png"
        out_image.save(image_out)
        out_ori_mask.save(ori_mask_out)
        out_tgt_mask.save(tgt_mask_out)

        coarse_input_path = ""
        coarse_src = str(row.get("coarse_input_path") or "")
        if coarse_src and Path(coarse_src).is_file():
            with Image.open(coarse_src) as coarse_img:
                coarse_out = resize_center_crop(coarse_img, size, size, is_mask=False)
            coarse_path = proc_coarse / f"{sample_id}.png"
            to_rgb(coarse_out).save(coarse_path)
            coarse_input_path = str(coarse_path)

        caption_4v = str(row.get("caption_4v") or "")
        sample = GeoBenchBenchmarkSample(
            sample_id=sample_id,
            ori_img_path=str(image_out),
            ori_mask_path=str(ori_mask_out),
            tgt_mask_path=str(tgt_mask_out),
            prompt=_resolve_prompt(row, prompt_field),
            caption_4v=caption_4v,
            edit_prompt=str(row.get("edit_prompt") or ""),
            edit_param_json=edit_param_json,
            obj_label=str(row.get("obj_label") or ""),
            coarse_input_path=coarse_input_path,
            edit_param_iou=edit_param_iou,
            target_width=size,
            target_height=size,
        )
        benchmark_rows.append(sample.to_dict())

    if iou_values:
        mean_iou = sum(iou_values) / len(iou_values)
        logger.info(
            "edit_param validation: mean IoU=%.3f, low IoU (<%s)=%d / %d",
            mean_iou,
            IOU_WARN_THRESHOLD,
            low_iou_count,
            len(iou_values),
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_dicts(out_csv, benchmark_rows, GEOBENCH_MANIFEST_FIELDS)
    logger.info("Wrote GeoBench manifest with %d rows to %s", len(benchmark_rows), out_csv)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GeoBench geometric-editing manifest CSV")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="eval/output/manifests/geobench_2d_1k.csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument(
        "--prompt-field",
        type=str,
        default=DEFAULT_GEOBENCH_PROMPT_FIELD,
        choices=["caption_4v", "edit_prompt"],
        help="Primary prompt column for the adapter (default: caption_4v)",
    )
    parser.add_argument("--processed-dir", type=str, default=None)
    args = parser.parse_args()

    build_geobench_manifest(
        resolve_path(args.input),
        resolve_path(args.out),
        seed=args.seed,
        size=args.size,
        prompt_field=args.prompt_field,
        processed_dir=resolve_path(args.processed_dir) if args.processed_dir else None,
    )


if __name__ == "__main__":
    main()
