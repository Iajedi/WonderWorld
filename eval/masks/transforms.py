"""Spatial transforms shared by images and masks."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from eval.utils import binarize_mask, mask_area_ratio, to_grayscale, to_rgb


def resize_center_crop(
    image: Image.Image,
    target_width: int,
    target_height: int,
    *,
    is_mask: bool = False,
    resample: int = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resize so the shortest side covers the target, then center-crop."""
    if is_mask:
        image = to_grayscale(image)
    else:
        image = to_rgb(image)

    src_w, src_h = image.size
    scale = max(target_width / src_w, target_height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = image.resize((new_w, new_h), resample=resample)
    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    cropped = resized.crop((left, top, left + target_width, top + target_height))

    if is_mask:
        return binarize_mask(cropped, auto_polarity=False)
    return cropped


def transform_image_and_mask(
    image: Image.Image,
    mask: Image.Image,
    target_width: int,
    target_height: int,
) -> tuple[Image.Image, Image.Image, float]:
    """Apply identical geometry to image and mask; return area ratio."""
    out_image = resize_center_crop(image, target_width, target_height, is_mask=False)
    out_mask = resize_center_crop(mask, target_width, target_height, is_mask=True)
    area = mask_area_ratio(out_mask)
    if area < 0.01:
        raise ValueError(f"Mask area ratio too small after transform: {area:.4f}")
    return out_image, out_mask, area


def transform_image_and_dual_masks(
    image: Image.Image,
    ori_mask: Image.Image,
    tgt_mask: Image.Image,
    target_width: int,
    target_height: int,
) -> tuple[Image.Image, Image.Image, Image.Image, float]:
    """Apply identical geometry to image and both masks; return ori_mask area ratio."""
    out_image = resize_center_crop(image, target_width, target_height, is_mask=False)
    out_ori_mask = resize_center_crop(ori_mask, target_width, target_height, is_mask=True)
    out_tgt_mask = resize_center_crop(tgt_mask, target_width, target_height, is_mask=True)
    area = mask_area_ratio(out_ori_mask)
    if area < 0.001:
        raise ValueError(f"Original mask area ratio too small after transform: {area:.4f}")
    return out_image, out_ori_mask, out_tgt_mask, area


def make_right_outpaint_mask(
    width: int,
    height: int,
    coverage_frac: float,
    *,
    blur_sigma: float = 10.0,
) -> Image.Image:
    """Build a soft right-side outpaint mask (white = hole) with blurred boundary."""
    if not 0.0 < coverage_frac < 1.0:
        raise ValueError(f"coverage_frac must be in (0, 1), got {coverage_frac}")

    arr = np.zeros((height, width), dtype=np.float32)
    split_x = int(round(width * (1.0 - coverage_frac)))
    split_x = min(max(split_x, 0), width)
    arr[:, split_x:] = 1.0

    if blur_sigma > 0:
        ksize = int(round(blur_sigma * 6)) | 1
        ksize = max(ksize, 3)
        arr = cv2.GaussianBlur(arr, (ksize, ksize), blur_sigma)

    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")
