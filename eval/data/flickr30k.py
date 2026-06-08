"""Flickr30k subset creation from Hugging Face datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from eval.config import (
    DEFAULT_CAPTION_STRATEGY,
    DEFAULT_FLICKR30K_DATASET,
    DEFAULT_FLICKR30K_SPLIT,
    DEFAULT_SUBSET_COUNT,
    DEFAULT_SUBSET_SEED,
    STREAMING_FLICKR30K_DATASET,
)
from eval.data.manifest import FlickrRecord, choose_caption
from eval.logging_utils import get_logger
from eval.paths import ensure_dir, resolve_path
from eval.seed import set_global_seed
from eval.utils import to_rgb, write_jsonl

logger = get_logger(__name__)


def _normalize_captions(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def _selected_indices(total: int, count: int, seed: int) -> list[int]:
    if count > total:
        raise ValueError(f"Requested count {count} exceeds dataset size {total}.")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total)
    return sorted(int(i) for i in perm[:count])


def _load_hf_dataset(
    dataset_name: str,
    split: str,
    *,
    streaming: bool,
):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": dataset_name, "split": split}
    if streaming:
        kwargs["streaming"] = True
    else:
        kwargs["trust_remote_code"] = True
    return load_dataset(**kwargs)


def _dataset_length(dataset_name: str, split: str) -> int:
    from datasets import load_dataset_builder

    kwargs: dict[str, Any] = {}
    if dataset_name == DEFAULT_FLICKR30K_DATASET:
        kwargs["trust_remote_code"] = True
    builder = load_dataset_builder(dataset_name, **kwargs)
    if split not in builder.info.splits:
        raise ValueError(f"Split '{split}' not found in dataset {dataset_name}")
    return int(builder.info.splits[split].num_examples)


def build_flickr30k_subset(
    out_dir: Path,
    *,
    count: int = DEFAULT_SUBSET_COUNT,
    seed: int = DEFAULT_SUBSET_SEED,
    dataset_name: str = DEFAULT_FLICKR30K_DATASET,
    split: str = DEFAULT_FLICKR30K_SPLIT,
    caption_strategy: str = DEFAULT_CAPTION_STRATEGY,
    streaming: bool | None = None,
) -> Path:
    """Create a fixed Flickr30k subset and JSONL manifest."""
    set_global_seed(seed)
    out_dir = ensure_dir(out_dir)
    images_dir = ensure_dir(out_dir / "images")

    use_streaming = streaming
    if use_streaming is None:
        use_streaming = dataset_name == STREAMING_FLICKR30K_DATASET

    total = _dataset_length(dataset_name, split)
    selected = set(_selected_indices(total, count, seed))
    logger.info(
        "Dataset=%s split=%s total=%d selected=%d streaming=%s",
        dataset_name,
        split,
        total,
        len(selected),
        use_streaming,
    )

    ds = _load_hf_dataset(dataset_name, split, streaming=use_streaming)
    records: list[dict[str, Any]] = []
    saved = 0

    if use_streaming:
        for idx, row in enumerate(ds):
            if idx not in selected:
                continue
            record = _save_row(
                row,
                idx,
                images_dir,
                sample_id=f"{saved:06d}",
                caption_strategy=caption_strategy,
                seed=seed,
            )
            records.append(record.to_dict())
            saved += 1
            if saved >= count:
                break
    else:
        for out_idx, ds_idx in enumerate(sorted(selected)):
            row = ds[int(ds_idx)]
            record = _save_row(
                row,
                int(ds_idx),
                images_dir,
                sample_id=f"{out_idx:06d}",
                caption_strategy=caption_strategy,
                seed=seed,
            )
            records.append(record.to_dict())

    if len(records) != count:
        raise RuntimeError(
            f"Expected {count} records but saved {len(records)}. "
            "Check dataset split/name or streaming settings."
        )

    manifest_path = out_dir / "flickr30k_1k.jsonl"
    write_jsonl(manifest_path, records)
    logger.info("Wrote %d records to %s", len(records), manifest_path)
    return manifest_path


def _save_row(
    row: dict[str, Any],
    dataset_index: int,
    images_dir: Path,
    *,
    sample_id: str | None = None,
    caption_strategy: str,
    seed: int,
) -> FlickrRecord:
    image = row.get("image")
    if image is None:
        raise ValueError(f"Row {dataset_index} has no image field.")

    if hasattr(image, "convert"):
        pil_image = to_rgb(image)
    else:
        from PIL import Image

        pil_image = to_rgb(Image.open(image))

    sid = sample_id or f"{len(list(images_dir.glob('*.png'))):06d}"
    image_path = images_dir / f"{sid}.png"
    pil_image.save(image_path)

    captions = _normalize_captions(row.get("caption"))
    prompt = choose_caption(
        captions,
        sample_id=sid,
        strategy=caption_strategy,
        seed=seed,
    )

    return FlickrRecord(
        sample_id=sid,
        split=str(row.get("split", "test")),
        dataset_index=dataset_index,
        img_id=str(row.get("img_id", row.get("image_id", sid))),
        filename=str(row.get("filename", image_path.name)),
        image_path=str(image_path),
        chosen_prompt=prompt,
        all_captions=captions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Flickr30k 1k subset from Hugging Face")
    parser.add_argument("--out", type=str, default="eval/output/flickr30k_1k")
    parser.add_argument("--count", type=int, default=DEFAULT_SUBSET_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--dataset", type=str, default=DEFAULT_FLICKR30K_DATASET)
    parser.add_argument("--split", type=str, default=DEFAULT_FLICKR30K_SPLIT)
    parser.add_argument(
        "--caption-strategy",
        type=str,
        default=DEFAULT_CAPTION_STRATEGY,
        choices=["first", "seeded_random"],
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Force streaming mode (recommended for lmms-lab/flickr30k)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming even for parquet datasets",
    )
    args = parser.parse_args()

    streaming = None
    if args.streaming:
        streaming = True
    elif args.no_streaming:
        streaming = False

    build_flickr30k_subset(
        resolve_path(args.out),
        count=args.count,
        seed=args.seed,
        dataset_name=args.dataset,
        split=args.split,
        caption_strategy=args.caption_strategy,
        streaming=streaming,
    )


if __name__ == "__main__":
    main()
