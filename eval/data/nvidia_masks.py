"""NVIDIA irregular mask testing-set retrieval."""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from eval.config import (
    DEFAULT_MASK_BINARIZE_THRESHOLD,
    DEFAULT_MASK_COUNT,
    DEFAULT_NVIDIA_BIN_INDICES,
    DEFAULT_SUBSET_SEED,
    NVIDIA_AREA_BINS,
    NVIDIA_MASKS_PER_BIN,
    NVIDIA_MASKS_PER_BIN_VARIANT,
    NVIDIA_OFFICIAL_PAGE_URL,
    NVIDIA_TEST_MASK_ZIP_URL,
)
from eval.data.manifest import MaskRecord
from eval.logging_utils import get_logger
from eval.paths import ensure_dir, resolve_path
from eval.seed import set_global_seed
from eval.utils import (
    binarize_mask,
    download_url,
    extract_tar_members,
    extract_zip_members,
    list_archive_mask_members,
    mask_area_ratio,
    resolve_download_url_from_page,
    write_jsonl,
)

logger = get_logger(__name__)

_TESTING_MASK_MEMBER_RE = re.compile(
    r"(?:^|/)testing_mask_dataset/(\d+)\.(?:png|jpg|jpeg|bmp)$",
    re.IGNORECASE,
)


def resolve_mask_archive_url(cli_url: str | None = None) -> str:
    """Resolve the mask archive download URL."""
    if cli_url:
        return cli_url
    if NVIDIA_TEST_MASK_ZIP_URL:
        return NVIDIA_TEST_MASK_ZIP_URL

    discovered = resolve_download_url_from_page(NVIDIA_OFFICIAL_PAGE_URL)
    if discovered:
        logger.info("Discovered archive URL from official page: %s", discovered)
        return discovered

    raise ValueError(
        "No NVIDIA testing mask archive URL configured. "
        "Pass --url with a direct link to the testing-set zip/tar, or set "
        "eval.config.NVIDIA_TEST_MASK_ZIP_URL. "
        f"Official page: {NVIDIA_OFFICIAL_PAGE_URL}"
    )


def area_bin_label(bin_index: int) -> str:
    """Return a human-readable label for a testing-set bin index."""
    if bin_index < 0 or bin_index >= len(NVIDIA_AREA_BINS):
        raise ValueError(f"Invalid NVIDIA bin index: {bin_index}")
    low, high = NVIDIA_AREA_BINS[bin_index]
    return f"({low:.2f}, {high:.2f}]"


def mask_index_from_member(member: str) -> int | None:
    """Parse the numeric testing-set mask index from an archive member path."""
    normalized = member.replace("\\", "/")
    match = _TESTING_MASK_MEMBER_RE.search(normalized)
    if not match:
        return None
    return int(match.group(1))


def bin_index_from_mask_index(mask_index: int) -> int:
    """Map a flat testing-set mask index to its area-ratio bin."""
    return mask_index // NVIDIA_MASKS_PER_BIN


def has_border_constraint(mask_index: int) -> bool:
    """Return True when the mask belongs to the border-constraint split."""
    return (mask_index % NVIDIA_MASKS_PER_BIN) >= NVIDIA_MASKS_PER_BIN_VARIANT


def parse_bin_indices(spec: str | None) -> tuple[int, ...]:
    """Parse comma-separated bin indices or area-bin labels."""
    if not spec:
        return DEFAULT_NVIDIA_BIN_INDICES

    indices: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            indices.append(int(token))
            continue

        normalized = token.replace(" ", "")
        for idx, (low, high) in enumerate(NVIDIA_AREA_BINS):
            labels = {
                f"({low},{high}]",
                f"({low:.2f},{high:.2f}]",
                f"{low}-{high}",
                f"{low:.2f}-{high:.2f}",
            }
            if normalized in labels:
                indices.append(idx)
                break
        else:
            raise ValueError(
                f"Unknown NVIDIA area bin '{token}'. "
                "Use bin indices like 0,1,2 or labels like 0.01-0.10."
            )

    if not indices:
        raise ValueError("No NVIDIA area bins selected.")
    return tuple(dict.fromkeys(indices))


def list_testing_mask_members(archive_path: Path) -> list[str]:
    """List testing-set mask members with parseable numeric indices."""
    members = [
        member
        for member in list_archive_mask_members(archive_path)
        if mask_index_from_member(member) is not None
    ]
    return sorted(members, key=lambda member: mask_index_from_member(member) or -1)


def filter_members_by_bins(
    members: list[str],
    bin_indices: tuple[int, ...],
) -> list[str]:
    """Keep only masks from the requested area-ratio bins."""
    allowed = set(bin_indices)
    filtered: list[str] = []
    for member in members:
        mask_index = mask_index_from_member(member)
        if mask_index is None:
            continue
        if bin_index_from_mask_index(mask_index) in allowed:
            filtered.append(member)
    return filtered


def _select_members(members: list[str], count: int, seed: int) -> list[str]:
    if count > len(members):
        raise ValueError(
            f"Requested {count} masks but only {len(members)} candidates "
            f"match the selected area bins."
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(members))
    return [members[int(i)] for i in perm[:count]]


def _flatten_extracted(path: Path, dest_dir: Path, mask_id: str) -> Path:
    """Move extracted file to flat dest_dir with stable name."""
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        if not files:
            raise ValueError(f"No files extracted for {mask_id}")
        src = files[0]
    else:
        src = path

    suffix = src.suffix.lower() or ".png"
    out_path = dest_dir / f"{mask_id}{suffix}"
    if src.resolve() != out_path.resolve():
        shutil.copy2(src, out_path)
    return out_path


def _clean_output_dir(out_dir: Path) -> None:
    """Remove previously materialized masks and manifest."""
    masks_dir = out_dir / "masks"
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    manifest_path = out_dir / "masks.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()


def build_nvidia_mask_subset(
    out_dir: Path,
    *,
    count: int = DEFAULT_MASK_COUNT,
    seed: int = DEFAULT_SUBSET_SEED,
    url: str | None = None,
    threshold: int = DEFAULT_MASK_BINARIZE_THRESHOLD,
    cache_dir: Path | None = None,
    bin_indices: tuple[int, ...] = DEFAULT_NVIDIA_BIN_INDICES,
    redownload: bool = False,
    clean: bool = True,
) -> Path:
    """Download testing masks and save a shuffled subset from selected area bins."""
    set_global_seed(seed)
    out_dir = ensure_dir(out_dir)
    if clean:
        _clean_output_dir(out_dir)
    masks_dir = ensure_dir(out_dir / "masks")
    cache = ensure_dir(cache_dir or out_dir / "cache")

    archive_url = resolve_mask_archive_url(url)
    archive_name = Path(archive_url.split("?")[0]).name or "nvidia_test_masks.zip"
    archive_path = cache / archive_name

    if redownload and archive_path.exists():
        logger.info("Removing cached archive %s", archive_path)
        archive_path.unlink()

    if not archive_path.exists() or archive_path.stat().st_size == 0:
        logger.info("Downloading mask archive from %s", archive_url)
        download_url(archive_url, archive_path, resume=False if redownload else True)
    else:
        logger.info("Using cached archive %s", archive_path)

    all_members = list_testing_mask_members(archive_path)
    if not all_members:
        raise RuntimeError(f"No testing-set mask files found in archive: {archive_path}")

    eligible_members = filter_members_by_bins(all_members, bin_indices)
    bin_labels = [area_bin_label(idx) for idx in bin_indices]
    logger.info(
        "Eligible masks from bins %s: %d/%d archive members",
        ", ".join(bin_labels),
        len(eligible_members),
        len(all_members),
    )
    if not eligible_members:
        raise RuntimeError(
            f"No masks matched bin indices {bin_indices}. "
            "Check --bins and archive layout."
        )

    selected_members = _select_members(eligible_members, count, seed)
    logger.info(
        "Selected %d shuffled masks from %d eligible candidates",
        len(selected_members),
        len(eligible_members),
    )

    records: list[dict] = []
    suffix = archive_path.suffix.lower()

    for idx, member in enumerate(selected_members):
        mask_id = f"{idx:06d}"
        source_index = mask_index_from_member(member)
        if source_index is None:
            raise RuntimeError(f"Could not parse mask index from member: {member}")

        extract_dir = cache / "extract" / mask_id
        extract_dir.mkdir(parents=True, exist_ok=True)

        if suffix == ".zip" or zipfile.is_zipfile(archive_path):
            extracted = extract_zip_members(archive_path, [member], extract_dir)
        else:
            extracted = extract_tar_members(archive_path, [member], extract_dir)

        raw_path = _flatten_extracted(extracted[0], masks_dir, mask_id)
        with Image.open(raw_path) as img:
            orig_w, orig_h = img.size
            binary = binarize_mask(img, threshold=threshold, auto_polarity=False)
            area = mask_area_ratio(binary)
            if area < 0.01:
                logger.warning("Skipping mask %s: area ratio %.4f too small", mask_id, area)
                continue
            final_path = masks_dir / f"{mask_id}.png"
            binary.save(final_path)

        if raw_path.exists() and raw_path != final_path:
            raw_path.unlink(missing_ok=True)

        bin_idx = bin_index_from_mask_index(source_index)
        record = MaskRecord(
            mask_id=mask_id,
            source_url=archive_url,
            source_member=member,
            mask_path=str(final_path),
            original_width=orig_w,
            original_height=orig_h,
            mask_area_ratio=area,
            area_bin=area_bin_label(bin_idx),
            border_constraint=has_border_constraint(source_index),
            source_index=source_index,
        )
        records.append(record.to_dict())

    if len(records) < count:
        raise RuntimeError(
            f"Saved {len(records)} masks but {count} were requested after filtering."
        )

    manifest_path = out_dir / "masks.jsonl"
    write_jsonl(manifest_path, records)
    logger.info("Wrote %d mask records to %s", len(records), manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NVIDIA irregular testing masks subset")
    parser.add_argument("--out", type=str, default="eval/output/nvidia_masks_test")
    parser.add_argument("--count", type=int, default=DEFAULT_MASK_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--url", type=str, default=None, help="Direct archive URL override")
    parser.add_argument(
        "--bins",
        type=str,
        default=",".join(str(i) for i in DEFAULT_NVIDIA_BIN_INDICES),
        help=(
            "Comma-separated area-bin indices (0-5) or labels like 0.01-0.10. "
            f"Default: {','.join(str(i) for i in DEFAULT_NVIDIA_BIN_INDICES)}"
        ),
    )
    parser.add_argument("--threshold", type=int, default=DEFAULT_MASK_BINARIZE_THRESHOLD)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Delete cached archive and download it again",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing masks/ in the output directory before writing",
    )
    args = parser.parse_args()

    build_nvidia_mask_subset(
        resolve_path(args.out),
        count=args.count,
        seed=args.seed,
        url=args.url,
        threshold=args.threshold,
        cache_dir=resolve_path(args.cache_dir) if args.cache_dir else None,
        bin_indices=parse_bin_indices(args.bins),
        redownload=args.redownload,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
