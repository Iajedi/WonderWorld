"""Build the final benchmark manifest by pairing images and masks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from eval.config import (
    DEFAULT_ABLATION_OUTPAINT_CSV,
    DEFAULT_ABLATION_OUTPAINT_DIR,
    DEFAULT_OUTPAINT_BLUR_SIGMA,
    DEFAULT_PANO_IMAGES_DIR,
    DEFAULT_PANO_MANIFEST,
    DEFAULT_PAIRING_MODE,
    DEFAULT_SUBSET_SEED,
    DEFAULT_TARGET_SIZE,
    DEFAULT_WORLDSCORE_STATIC_ROOT,
    OUTPAINT_COVERAGE_LEVELS,
)
from eval.data.manifest import (
    ABLATION_OUTPAINT_FIELDS,
    AblationOutpaintSample,
    BenchmarkSample,
    captions_to_json,
)
from eval.logging_utils import get_logger
from eval.masks.pairing import pair_indices
from eval.masks.transforms import make_right_outpaint_mask, transform_image_and_mask
from eval.paths import ensure_dir, resolve_path
from eval.seed import set_global_seed
from eval.utils import compose_black_inpaint, read_jsonl, to_rgb, write_csv_dicts

logger = get_logger(__name__)

BENCHMARK_FIELDS = [
    "sample_id",
    "image_path",
    "prompt",
    "all_captions_json",
    "mask_id",
    "mask_path",
    "target_width",
    "target_height",
]


def _clean_ablation_dir(processed_dir: Path) -> None:
    for subdir in ("images", "masks", "composed"):
        path = processed_dir / subdir
        if path.exists():
            shutil.rmtree(path)


def _resolve_pano_image_path(scene_id: str, pano_images_dir: Path) -> Path:
    image_path = pano_images_dir / f"{scene_id}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Pano image not found for scene_id={scene_id}: {image_path}")
    return image_path


def _load_next_scene_prompt(notes: str, worldscore_root: Path) -> str:
    """Return the first prompt from the sibling 001 scene directory."""
    notes_path = notes.strip().rstrip("/")
    if notes_path.endswith("/000"):
        next_notes = notes_path[:-4] + "/001"
    elif notes_path.endswith("000"):
        next_notes = notes_path[:-3] + "001"
    else:
        raise ValueError(f"Expected notes path ending in 000, got: {notes!r}")

    image_data_path = worldscore_root / next_notes / "image_data.json"
    if not image_data_path.is_file():
        raise FileNotFoundError(f"Next-scene image_data.json not found: {image_data_path}")

    with open(image_data_path, encoding="utf-8") as f:
        data = json.load(f)

    prompt_list = data.get("prompt_list", [])
    if not prompt_list:
        raise ValueError(f"No prompt_list in {image_data_path}")
    return str(prompt_list[0])


def _build_scene_prompt_map(pano_manifest: Path, worldscore_root: Path) -> dict[str, tuple[str, str]]:
    pano_df = pd.read_csv(pano_manifest, dtype={"scene_id": str})
    prompt_map: dict[str, tuple[str, str]] = {}
    for _, scene in pano_df.iterrows():
        scene_id = str(scene["scene_id"])
        first_prompt = str(scene["prompt"])
        notes = str(scene.get("notes", ""))
        if not notes:
            raise ValueError(f"scene_id={scene_id} missing notes column in {pano_manifest}")
        next_prompt = _load_next_scene_prompt(notes, worldscore_root)
        prompt_map[scene_id] = (first_prompt, next_prompt)
    return prompt_map


def _backfill_first_prompt_fields(row: dict) -> dict:
    prompt = str(row["prompt"])
    out = dict(row)
    out["prompt_src"] = str(out.get("prompt_src") or prompt)
    out["prompt_tgt"] = str(out.get("prompt_tgt") or prompt)
    out["prompt_variant"] = str(out.get("prompt_variant") or "first")
    return out


def expand_ablation_outpaint_next_prompts(
    existing_csv: Path,
    pano_manifest: Path,
    worldscore_root: Path,
    out_csv: Path,
) -> Path:
    """Append next-scene prompt rows to an existing ablation outpaint manifest."""
    existing_df = pd.read_csv(existing_csv, dtype={"sample_id": str, "mask_id": str, "scene_id": str})
    if existing_df.empty:
        raise ValueError(f"No rows found in {existing_csv}")

    if "prompt_variant" in existing_df.columns:
        first_rows = existing_df[existing_df["prompt_variant"].fillna("first") == "first"].copy()
        if len(first_rows) != len(existing_df):
            raise ValueError(
                f"{existing_csv} already contains next-scene rows; "
                "run expansion only on a first-prompt-only manifest."
            )
    else:
        first_rows = existing_df.copy()

    prompt_map = _build_scene_prompt_map(pano_manifest, worldscore_root)
    first_rows_list = [_backfill_first_prompt_fields(row) for row in first_rows.to_dict("records")]

    next_rows: list[dict] = []
    next_idx = len(first_rows_list)
    for row in first_rows_list:
        scene_id = str(row["scene_id"])
        if scene_id not in prompt_map:
            raise KeyError(f"scene_id={scene_id} not found in {pano_manifest}")
        first_prompt, next_prompt = prompt_map[scene_id]

        sample_id = f"{next_idx:06d}"
        next_rows.append(
            {
                **row,
                "sample_id": sample_id,
                "mask_id": sample_id,
                "prompt": next_prompt,
                "all_captions_json": captions_to_json([first_prompt, next_prompt]),
                "prompt_src": first_prompt,
                "prompt_tgt": next_prompt,
                "prompt_variant": "next",
            }
        )
        next_idx += 1

    all_rows = first_rows_list + next_rows
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_dicts(out_csv, all_rows, ABLATION_OUTPAINT_FIELDS)
    logger.info(
        "Expanded outpaint ablation manifest: %d first rows + %d next rows -> %s",
        len(first_rows_list),
        len(next_rows),
        out_csv,
    )
    return out_csv


def build_ablation_outpaint_manifest(
    pano_manifest: Path,
    pano_images_dir: Path,
    out_csv: Path,
    *,
    processed_dir: Path | None = None,
    size: int = DEFAULT_TARGET_SIZE,
    coverages: tuple[float, ...] = OUTPAINT_COVERAGE_LEVELS,
    blur_sigma: float = DEFAULT_OUTPAINT_BLUR_SIGMA,
    clean: bool = True,
) -> Path:
    """Build the right-side outpaint ablation manifest from processed pano scenes."""
    pano_df = pd.read_csv(pano_manifest, dtype={"scene_id": str})
    if pano_df.empty:
        raise ValueError(f"No pano scenes found in {pano_manifest}")

    base_processed = ensure_dir(processed_dir or resolve_path(DEFAULT_ABLATION_OUTPAINT_DIR))
    if clean:
        _clean_ablation_dir(base_processed)

    proc_images = ensure_dir(base_processed / "images")
    proc_masks = ensure_dir(base_processed / "masks")
    proc_composed = ensure_dir(base_processed / "composed")

    rows: list[dict] = []
    sample_idx = 0

    for _, scene in pano_df.iterrows():
        scene_id = str(scene["scene_id"])
        prompt = str(scene["prompt"])
        source_image = _resolve_pano_image_path(scene_id, pano_images_dir)

        with Image.open(source_image) as img:
            rgb = to_rgb(img).copy()
            if rgb.size != (size, size):
                rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)

        for coverage in coverages:
            sample_id = f"{sample_idx:06d}"
            mask = make_right_outpaint_mask(size, size, coverage, blur_sigma=blur_sigma)
            composed = compose_black_inpaint(rgb, mask)

            image_out = proc_images / f"{sample_id}.png"
            mask_out = proc_masks / f"{sample_id}.png"
            composed_out = proc_composed / f"{sample_id}.png"

            rgb.save(image_out)
            mask.save(mask_out)
            composed.save(composed_out)

            sample = AblationOutpaintSample(
                sample_id=sample_id,
                image_path=str(image_out.resolve()),
                composed_path=str(composed_out.resolve()),
                prompt=prompt,
                all_captions_json=captions_to_json([prompt]),
                mask_id=sample_id,
                mask_path=str(mask_out.resolve()),
                target_width=size,
                target_height=size,
                scene_id=scene_id,
                outpaint_coverage=float(coverage),
                prompt_src=prompt,
                prompt_tgt=prompt,
                prompt_variant="first",
            )
            rows.append(sample.to_dict())
            sample_idx += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_dicts(out_csv, rows, ABLATION_OUTPAINT_FIELDS)
    logger.info(
        "Wrote outpaint ablation manifest with %d rows (%d scenes x %d coverages) to %s",
        len(rows),
        len(pano_df),
        len(coverages),
        out_csv,
    )
    return out_csv


def build_benchmark_manifest(
    images_manifest: Path,
    masks_manifest: Path,
    out_csv: Path,
    *,
    seed: int = DEFAULT_SUBSET_SEED,
    size: int = DEFAULT_TARGET_SIZE,
    pairing_mode: str = DEFAULT_PAIRING_MODE,
    processed_dir: Path | None = None,
) -> Path:
    """Pair Flickr images with NVIDIA masks and write benchmark CSV."""
    set_global_seed(seed)

    image_rows = read_jsonl(images_manifest)
    mask_rows = read_jsonl(masks_manifest)
    if not image_rows:
        raise ValueError(f"No image records found in {images_manifest}")
    if not mask_rows:
        raise ValueError(f"No mask records found in {masks_manifest}")

    mask_indices = pair_indices(
        len(image_rows),
        len(mask_rows),
        mode=pairing_mode,
        seed=seed,
    )

    base_processed = ensure_dir(processed_dir or out_csv.parent / "processed_flickr")
    proc_images = ensure_dir(base_processed / "images")
    proc_masks = ensure_dir(base_processed / "masks")

    benchmark_rows: list[dict] = []

    for img_idx, mask_idx in enumerate(mask_indices):
        img_row = image_rows[img_idx]
        mask_row = mask_rows[mask_idx]

        sample_id = str(img_row.get("sample_id", f"{img_idx:06d}"))
        mask_id = str(mask_row.get("mask_id", f"{mask_idx:06d}"))

        with Image.open(img_row["image_path"]) as img, Image.open(mask_row["mask_path"]) as msk:
            out_image, out_mask, _ = transform_image_and_mask(img, msk, size, size)

        image_out = proc_images / f"{sample_id}.png"
        mask_out = proc_masks / f"{sample_id}.png"
        out_image.save(image_out)
        out_mask.save(mask_out)

        captions = img_row.get("all_captions", [])
        if isinstance(captions, str):
            all_captions_json = captions
        else:
            all_captions_json = captions_to_json(list(captions))

        sample = BenchmarkSample(
            sample_id=sample_id,
            image_path=str(image_out),
            prompt=str(img_row.get("chosen_prompt", "")),
            all_captions_json=all_captions_json,
            mask_id=mask_id,
            mask_path=str(mask_out),
            target_width=size,
            target_height=size,
        )
        benchmark_rows.append(sample.to_dict())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_dicts(out_csv, benchmark_rows, BENCHMARK_FIELDS)
    logger.info("Wrote benchmark manifest with %d rows to %s", len(benchmark_rows), out_csv)
    return out_csv


def _parse_coverages(spec: str | None) -> tuple[float, ...]:
    if not spec:
        return OUTPAINT_COVERAGE_LEVELS
    values = tuple(float(part.strip()) for part in spec.split(",") if part.strip())
    if not values:
        raise ValueError("No coverage values parsed from --coverages.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark manifest CSV")
    parser.add_argument(
        "--ablation-outpaint",
        action="store_true",
        help="Build right-side outpaint ablation manifest from processed pano scenes",
    )
    parser.add_argument("--images-manifest", type=str, default=None)
    parser.add_argument("--masks-manifest", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument(
        "--pairing",
        type=str,
        default=DEFAULT_PAIRING_MODE,
        choices=["fixed", "random"],
    )
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument(
        "--pano-manifest",
        type=str,
        default=DEFAULT_PANO_MANIFEST,
        help="Processed pano manifest CSV (used with --ablation-outpaint)",
    )
    parser.add_argument(
        "--pano-images-dir",
        type=str,
        default=DEFAULT_PANO_IMAGES_DIR,
        help="Directory containing processed pano PNGs",
    )
    parser.add_argument(
        "--coverages",
        type=str,
        default=None,
        help="Comma-separated right-side coverage fractions, e.g. 0.05,0.15,...,0.95",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=DEFAULT_OUTPAINT_BLUR_SIGMA,
        help="Gaussian blur sigma for the outpaint mask boundary",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing ablation_outpaint images/masks/composed before building",
    )
    parser.add_argument(
        "--expand-next-prompt",
        action="store_true",
        help="Append next-scene prompt rows to an existing ablation outpaint manifest",
    )
    parser.add_argument(
        "--worldscore-root",
        type=str,
        default=DEFAULT_WORLDSCORE_STATIC_ROOT,
        help="Root directory containing worldscore static scene data (used with --expand-next-prompt)",
    )
    args = parser.parse_args()

    if args.expand_next_prompt:
        expand_ablation_outpaint_next_prompts(
            resolve_path(args.out or DEFAULT_ABLATION_OUTPAINT_CSV),
            resolve_path(args.pano_manifest),
            resolve_path(args.worldscore_root),
            resolve_path(args.out or DEFAULT_ABLATION_OUTPAINT_CSV),
        )
        return

    if args.ablation_outpaint:
        build_ablation_outpaint_manifest(
            resolve_path(args.pano_manifest),
            resolve_path(args.pano_images_dir),
            resolve_path(args.out or DEFAULT_ABLATION_OUTPAINT_CSV),
            processed_dir=resolve_path(args.processed_dir or DEFAULT_ABLATION_OUTPAINT_DIR),
            size=args.size,
            coverages=_parse_coverages(args.coverages),
            blur_sigma=args.blur_sigma,
            clean=args.clean,
        )
        return

    if not args.images_manifest or not args.masks_manifest:
        parser.error("--images-manifest and --masks-manifest are required unless --ablation-outpaint is set.")

    build_benchmark_manifest(
        resolve_path(args.images_manifest),
        resolve_path(args.masks_manifest),
        resolve_path(args.out or "eval/output/manifests/benchmark_1k.csv"),
        seed=args.seed,
        size=args.size,
        pairing_mode=args.pairing,
        processed_dir=resolve_path(args.processed_dir) if args.processed_dir else None,
    )


if __name__ == "__main__":
    main()
