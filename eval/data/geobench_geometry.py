"""GeoBench edit_param parsing and mask transform utilities."""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from eval.utils import binarize_mask, to_grayscale, to_rgb


EDIT_PARAM_NAMES = (
    "edit_dx",
    "edit_dy",
    "edit_dz",
    "edit_rx",
    "edit_ry",
    "edit_rz",
    "edit_sx",
    "edit_sy",
    "edit_sz",
)


def parse_edit_param(raw: str | list[Any] | None) -> tuple[float, ...]:
    """Parse GeoBench edit_param into nine floats."""
    if raw is None or raw == "":
        raise ValueError("edit_param is missing or empty.")
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = raw
    if not isinstance(data, (list, tuple)) or len(data) != 9:
        raise ValueError(f"edit_param must be a 9-element list, got {data!r}")
    return tuple(float(x) for x in data)


def edit_param_dict(raw: str | list[Any] | None) -> dict[str, float]:
    """Return edit_param as a named dict for CSV/JSONL export."""
    values = parse_edit_param(raw)
    return dict(zip(EDIT_PARAM_NAMES, values))


EDIT_PARAM_TYPE_NAMES = ("translation", "rotation", "scaling", "affine")


def classify_edit_param_type(raw: str | list[Any] | None) -> str:
    """Classify a GeoBench edit_param into translation, rotation, scaling, or affine.

    Uses the same parameter layout as :func:`edit_param_to_cv2_affine`:
    translation changes only ``dx``/``dy``; rotation only ``rz``; scaling only
    ``sx``/``sy``; all other non-identity transforms are affine.
    """
    dx, dy, dz, rx, ry, rz, sx, sy, sz = parse_edit_param(raw)
    eps = 1e-6

    def is_zero(value: float) -> bool:
        return abs(value) < eps

    def is_one(value: float) -> bool:
        return abs(value - 1.0) < eps

    others_zero = is_zero(dz) and is_zero(rx) and is_zero(ry)
    trans_default = is_zero(rz) and is_one(sx) and is_one(sy) and is_one(sz)
    rot_default = is_zero(dx) and is_zero(dy) and is_one(sx) and is_one(sy) and is_one(sz)
    scale_default = is_zero(dx) and is_zero(dy) and is_zero(rz)

    if (not is_zero(dx) or not is_zero(dy)) and others_zero and trans_default:
        return "translation"
    if not is_zero(rz) and others_zero and rot_default:
        return "rotation"
    if (not is_one(sx) or not is_one(sy)) and others_zero and scale_default and is_one(sz):
        return "scaling"
    return "affine"


def _mask_centroid(mask_arr: np.ndarray) -> tuple[float, float]:
    """Return (cx, cy) of foreground pixels in a binary mask array."""
    ys, xs = np.where(mask_arr >= 128)
    if len(xs) == 0:
        h, w = mask_arr.shape[:2]
        return (w - 1) * 0.5, (h - 1) * 0.5
    return float(xs.mean()), float(ys.mean())


def edit_param_to_cv2_affine(
    ori_mask: np.ndarray | torch.Tensor | Image.Image,
    edit_param: str | list[Any] | tuple[float, ...],
) -> np.ndarray:
    """Return 2x3 OpenCV affine matrix (FreeFine ``re_edit_2d`` convention)."""
    dx, dy, _dz, _rx, _ry, rz, sx, sy, _sz = parse_edit_param(edit_param)

    if isinstance(ori_mask, Image.Image):
        mask_arr = np.asarray(binarize_mask(to_grayscale(ori_mask)), dtype=np.uint8)
    elif isinstance(ori_mask, torch.Tensor):
        mask_arr = (ori_mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[0]
    else:
        mask_arr = np.asarray(ori_mask, dtype=np.uint8)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

    cx, cy = _mask_centroid(mask_arr)
    sx, sy = float(sx), float(sy)

    m23 = cv2.getRotationMatrix2D((cx, cy), -float(rz), 1.0)
    tx = cx * (1.0 - sx) + dx
    ty = cy * (1.0 - sy) + dy
    m23[0, 2] += tx
    m23[1, 2] += ty
    m23[0, 0] *= sx
    m23[0, 1] *= sx
    m23[1, 0] *= sy
    m23[1, 1] *= sy
    return m23.astype(np.float32)


def apply_edit_param_to_image(
    ori_img: Image.Image,
    ori_mask: Image.Image,
    edit_param: str | list[Any] | tuple[float, ...],
    *,
    size: int = 512,
    background: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image]:
    """Warp the object with ``edit_param`` and composite onto ``background``.

    Returns ``(tgt_image, warped_object_mask)`` where ``tgt_image`` matches
    FreeFine ``re_edit_2d``: warped object pixels replace the background only
    inside the transformed object mask.
    """
    rgb = to_rgb(ori_img).resize((size, size), Image.Resampling.LANCZOS)
    mask = binarize_mask(
        to_grayscale(ori_mask).resize((size, size), Image.Resampling.NEAREST)
    )
    bg = to_rgb(background).resize((size, size), Image.Resampling.LANCZOS) if background else rgb

    img_arr = np.asarray(rgb, dtype=np.uint8)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    bg_arr = np.asarray(bg, dtype=np.uint8)
    wh = (size, size)

    affine = edit_param_to_cv2_affine(mask, edit_param)
    warped_img = cv2.warpAffine(img_arr, affine, wh, flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(mask_arr, affine, wh, flags=cv2.INTER_NEAREST)
    active = warped_mask >= 128

    out_arr = np.where(active[..., None], warped_img, bg_arr)
    tgt_image = Image.fromarray(out_arr.astype(np.uint8), mode="RGB")
    warped_mask_pil = Image.fromarray((active.astype(np.uint8) * 255), mode="L")
    return tgt_image, warped_mask_pil


def verify_tgt_image_mask_adherence(
    warped_object_mask: Image.Image,
    tgt_mask: Image.Image,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | bool]:
    """Check that the edit_param-warped object mask aligns with ``tgt_mask``."""
    iou = mask_iou(warped_object_mask, tgt_mask)
    return {
        "warped_mask_tgt_iou": iou,
        "valid": bool(iou >= iou_threshold),
        "iou_threshold": iou_threshold,
    }


def edit_param_to_transform_matrix(
    ori_mask: np.ndarray | torch.Tensor,
    edit_param: str | list[Any] | tuple[float, ...],
) -> torch.Tensor:
    """Build a 3x3 forward affine matrix from GeoBench edit_param (FreeFine-style).

    Convention matches :func:`backbone.geometry.spec.affine_transform_mask`:
    column vectors ``[j; i; 1]`` with column ``j``, row ``i``.
    """
    if isinstance(ori_mask, torch.Tensor):
        mask_arr = (ori_mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[0]
    else:
        mask_arr = np.asarray(ori_mask, dtype=np.uint8)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

    m23 = edit_param_to_cv2_affine(mask_arr, edit_param)

    # OpenCV (x,y) maps to our (j,i) column/row convention.
    mat = np.array(
        [
            [m23[0, 0], m23[0, 1], m23[0, 2]],
            [m23[1, 0], m23[1, 1], m23[1, 2]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return torch.tensor(mat, dtype=torch.float32)


def _mask_to_tensor(mask: Image.Image, size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    w, h = size
    m = binarize_mask(to_grayscale(mask).resize((w, h), Image.Resampling.NEAREST))
    arr = (np.asarray(m, dtype=np.float32) / 255.0)
    return torch.from_numpy(arr).unsqueeze(0)


def predict_tgt_mask_from_edit_param(
    ori_mask: Image.Image,
    edit_param: str | list[Any] | tuple[float, ...],
    *,
    size: int = 512,
) -> Image.Image:
    """Warp ``ori_mask`` with ``edit_param`` and return a binarized PIL mask."""
    tensor = _mask_to_tensor(ori_mask, (size, size))
    mask_arr = (tensor.squeeze(0).numpy() >= 0.5).astype(np.uint8) * 255
    affine = edit_param_to_cv2_affine(mask_arr, edit_param)
    warped_arr = cv2.warpAffine(mask_arr, affine, (size, size), flags=cv2.INTER_NEAREST)
    arr = (warped_arr >= 128).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def mask_iou(mask_a: Image.Image, mask_b: Image.Image) -> float:
    """Intersection-over-union between two binarized masks."""
    a = np.asarray(binarize_mask(to_grayscale(mask_a)), dtype=bool)
    b = np.asarray(binarize_mask(to_grayscale(mask_b)), dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Mask shapes must match for IoU, got {a.shape} vs {b.shape}")
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)
