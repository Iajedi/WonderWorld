"""Shared utilities for the evaluation suite."""

from __future__ import annotations

import csv
import json
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from PIL import Image

from eval.logging_utils import get_logger

logger = get_logger(__name__)

MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def download_url(
    url: str,
    dest: Path,
    *,
    resume: bool = True,
    chunk_size: int = 1 << 20,
    timeout: int = 60,
) -> Path:
    """Download a URL to ``dest`` with optional resume support."""
    if "dropbox.com" in url and "dl=0" in url:
        url = url.replace("dl=0", "dl=1")
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {}
    mode = "wb"
    downloaded = 0

    if resume and dest.exists():
        downloaded = dest.stat().st_size
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"
            mode = "ab"

    with requests.get(url, stream=True, headers=headers, timeout=timeout) as resp:
        if resp.status_code == 416:
            logger.info("File already complete: %s", dest)
            return dest
        if resp.status_code not in (200, 206):
            resp.raise_for_status()

        with open(dest, mode) as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return dest


def extract_zip_members(
    archive_path: Path,
    members: Iterable[str],
    dest_dir: Path,
) -> list[Path]:
    """Extract selected members from a zip archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in members:
            zf.extract(member, path=dest_dir)
            extracted.append(dest_dir / member)
    return extracted


def extract_tar_members(
    archive_path: Path,
    members: Iterable[str],
    dest_dir: Path,
) -> list[Path]:
    """Extract selected members from a tar archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(archive_path, "r:*") as tf:
        for member in members:
            tf.extract(member, path=dest_dir)
            extracted.append(dest_dir / member)
    return extracted


def list_archive_mask_members(archive_path: Path) -> list[str]:
    """List mask-like members inside a zip or tar archive."""
    suffix = archive_path.suffix.lower()
    members: list[str] = []

    if suffix == ".zip" or zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for name in zf.namelist():
                if _is_mask_member(name):
                    members.append(name)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.isfile() and _is_mask_member(member.name):
                    members.append(member.name)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    return sorted(members)


def _is_mask_member(name: str) -> bool:
    lower = name.lower()
    if "/train" in lower or "\\train" in lower:
        return False
    return Path(lower).suffix in MASK_EXTENSIONS


def resolve_download_url_from_page(page_url: str) -> str | None:
    """Try to extract a mask archive URL from the official NVIDIA page."""
    resp = requests.get(page_url, timeout=30)
    resp.raise_for_status()
    html = resp.text

    patterns = [
        r'href="([^"]+\.zip[^"]*)"',
        r'href="([^"]+\.tar\.gz[^"]*)"',
        r'href="([^"]+\.tgz[^"]*)"',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, html, flags=re.IGNORECASE):
            if "test" in match.lower() and "mask" in match.lower():
                return urljoin(page_url, match.replace("dl=0", "dl=1"))
    for pattern in patterns:
        for match in re.findall(pattern, html, flags=re.IGNORECASE):
            if "mask" in match.lower() or "test" in match.lower():
                return urljoin(page_url, match.replace("dl=0", "dl=1"))
    return None


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert a PIL image to RGB."""
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def to_grayscale(image: Image.Image) -> Image.Image:
    """Convert a PIL image to single-channel grayscale."""
    if image.mode == "L":
        return image
    return image.convert("L")


def binarize_mask(
    mask: Image.Image,
    *,
    threshold: int = 127,
    white_is_hole: bool = True,
    auto_polarity: bool = False,
) -> Image.Image:
    """Binarize a mask so white means editable/inpaint region."""
    gray = to_grayscale(mask)
    arr = np.array(gray, dtype=np.uint8)
    binary = (arr >= threshold).astype(np.uint8) * 255

    if auto_polarity:
        white_ratio = float(binary.mean()) / 255.0
        if white_ratio < 0.5:
            binary = 255 - binary

    if not white_is_hole:
        binary = 255 - binary

    return Image.fromarray(binary, mode="L")


def mask_area_ratio(mask: Image.Image) -> float:
    """Return fraction of pixels that are inpaint (white) regions."""
    arr = np.array(to_grayscale(mask), dtype=np.float32) / 255.0
    return float((arr >= 0.5).mean())


def masked_pixel_arrays(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened RGB arrays for masked pixels only."""
    ref_arr = np.array(to_rgb(ref), dtype=np.float32)
    pred_arr = np.array(to_rgb(pred), dtype=np.float32)
    mask_arr = np.array(to_grayscale(mask), dtype=np.float32) / 255.0

    if ref_arr.shape[:2] != pred_arr.shape[:2]:
        raise ValueError("Reference and prediction sizes must match.")
    if mask_arr.shape[:2] != ref_arr.shape[:2]:
        raise ValueError("Mask size must match image size.")

    hole = mask_arr >= 0.5
    if hole.sum() == 0:
        raise ValueError("Mask has no inpaint region.")

    ref_pixels = ref_arr[hole]
    pred_pixels = pred_arr[hole]
    return ref_pixels, pred_pixels


def compose_black_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Composite an image with black pixels in the masked (white) region."""
    ref = np.array(to_rgb(image), dtype=np.float32)
    m = np.array(to_grayscale(mask), dtype=np.float32) / 255.0
    m = m[..., None]
    out = ref * (1.0 - m)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def composite_for_masked_metric(
    ref: Image.Image,
    pred: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Composite prediction with reference outside the inpaint region."""
    ref_rgb = to_rgb(ref)
    pred_rgb = to_rgb(pred)
    mask_l = binarize_mask(mask)

    ref_arr = np.array(ref_rgb, dtype=np.float32)
    pred_arr = np.array(pred_rgb, dtype=np.float32)
    m = np.array(mask_l, dtype=np.float32) / 255.0
    m = m[..., None]

    out = pred_arr * m + ref_arr * (1.0 - m)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def mask_bounding_box(mask: Image.Image) -> tuple[int, int, int, int]:
    """Return (left, upper, right, lower) bounding box of inpaint region."""
    arr = np.array(to_grayscale(mask), dtype=np.uint8)
    ys, xs = np.where(arr >= 128)
    if len(xs) == 0:
        raise ValueError("Mask has no inpaint region.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dict rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dicts."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write dict rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_csv_dict_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    """Append one row to CSV, writing the header when the file is new or empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def iter_with_progress(items: Iterable[Any], total: int | None, desc: str) -> Iterator[Any]:
    """Wrap an iterable with tqdm when available."""
    try:
        from tqdm import tqdm

        return iter(tqdm(items, total=total, desc=desc))
    except ImportError:
        return iter(items)
