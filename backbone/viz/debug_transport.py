"""Debug visualisation utilities for BCOT-HVE warm-start.

All functions are fire-and-forget: they write images / tensors to disk and
return ``None``.  Errors are caught and printed so they never block the
main pipeline.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _safe(fn):
    """Decorator that catches and prints exceptions instead of raising."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"[viz] {fn.__name__} failed: {e}")
    return wrapper


@_safe
def save_transport_map(
    pi: torch.Tensor,
    mask_2d: torch.Tensor,
    token_hw: Tuple[int, int],
    path: str,
) -> None:
    """Visualise the OT transport plan as a heatmap.

    Saves a per-unknown-token image showing how much weight each known
    token receives.  The aggregate (sum over unknown queries) is saved as
    a single-channel PNG.

    Parameters
    ----------
    pi : ``[B, U, K]``  transport plan.
    mask_2d : ``[H, W]``  binary mask.
    token_hw : ``(H, W)`` token grid dimensions.
    path : str  output file path (PNG).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    H, W = token_hw
    pi_cpu = pi[0].float().cpu()  # [U, K]

    known_flat = (mask_2d.reshape(-1) < 0.5).cpu()
    kn_idx = known_flat.nonzero(as_tuple=False).squeeze(-1)

    agg_weight = pi_cpu.sum(dim=0)  # [K]
    canvas = torch.zeros(H * W, dtype=torch.float32)
    n = min(kn_idx.numel(), agg_weight.numel())
    canvas[kn_idx[:n]] = agg_weight[:n]
    canvas = canvas.reshape(H, W)
    canvas = canvas / (canvas.max() + 1e-8)
    arr = (canvas.numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@_safe
def save_boundary_velocity_map(
    v_bdry: torch.Tensor,
    mask_2d: torch.Tensor,
    token_hw: Tuple[int, int],
    path: str,
) -> None:
    """Save the norm of boundary-token velocities as a heatmap.

    Parameters
    ----------
    v_bdry : ``[B, B_n, C]``
    mask_2d : ``[H, W]``
    token_hw : ``(H, W)``
    path : str
    """
    try:
        from ..utils.mask_ops import get_boundary_token_indices
    except ImportError:
        from backbone.utils.mask_ops import get_boundary_token_indices
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    H, W = token_hw
    bdry_idx = get_boundary_token_indices(mask_2d, width=2).cpu()

    norms = v_bdry[0].float().norm(dim=-1).cpu()  # [B_n]
    canvas = torch.zeros(H * W, dtype=torch.float32)
    if bdry_idx.numel() > 0 and norms.numel() > 0:
        canvas[bdry_idx[: norms.numel()]] = norms[: bdry_idx.numel()]
    canvas = canvas.reshape(H, W)
    canvas = canvas / (canvas.max() + 1e-8)
    arr = (canvas.numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@_safe
def save_latent_comparison(
    z_before: torch.Tensor,
    z_after: torch.Tensor,
    mask_token: torch.Tensor,
    token_hw: Tuple[int, int],
    path: str,
) -> None:
    """Save side-by-side latent norm maps (before / after warm-start).

    Parameters
    ----------
    z_before, z_after : ``[2, N, C]``  packed latents.
    mask_token : ``[1, N, 1]``
    token_hw : ``(H, W)``
    path : str
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    H, W = token_hw

    def _norm_map(z: torch.Tensor) -> np.ndarray:
        norms = z[0].float().norm(dim=-1).cpu().reshape(H, W)
        norms = norms / (norms.max() + 1e-8)
        return (norms.numpy() * 255).astype(np.uint8)

    before_img = _norm_map(z_before)
    after_img = _norm_map(z_after)
    mask_vis = (mask_token[0, :, 0].float().cpu().reshape(H, W).numpy() * 255).astype(np.uint8)

    combined = np.concatenate([before_img, after_img, mask_vis], axis=1)
    Image.fromarray(combined, mode="L").save(path)


@_safe
def save_seam_diagnostic(
    image: Image.Image,
    mask_np: np.ndarray,
    path: str,
) -> None:
    """Save gradient-magnitude map in a band around the mask boundary.

    Parameters
    ----------
    image : decoded PIL output image.
    mask_np : ``[1, 1, H, W]`` or ``[H, W]`` mask array.
    path : str
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = np.array(image.resize((512, 512))).astype(np.float32) / 255.0
    gray = np.mean(arr, axis=-1)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)
    vis = (grad_mag * 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(path)


@_safe
def save_comparison_grid(
    images_dict: Dict[str, Image.Image],
    path: str,
    target_size: Tuple[int, int] = (512, 512),
) -> None:
    """Tile multiple result images into a labelled comparison grid.

    Parameters
    ----------
    images_dict : mapping from method name to PIL image.
    path : str  output file path (PNG).
    target_size : size to which each image is resized.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    names = list(images_dict.keys())
    imgs = [images_dict[n].resize(target_size) for n in names]
    n = len(imgs)
    if n == 0:
        return
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    w, h = target_size
    label_h = 30
    canvas = Image.new("RGB", (cols * w, rows * (h + label_h)), (255, 255, 255))

    for i, (name, img) in enumerate(zip(names, imgs)):
        r, c = divmod(i, cols)
        canvas.paste(img, (c * w, r * (h + label_h) + label_h))
        # simple label (no font dependency)
        for x in range(w):
            for dy in range(min(2, label_h)):
                canvas.putpixel((c * w + x, r * (h + label_h) + dy), (200, 200, 200))

    canvas.save(path)
    _write_labels_txt(path.replace(".png", "_labels.txt"), names)


def _write_labels_txt(path: str, names: list) -> None:
    with open(path, "w") as f:
        for i, n in enumerate(names):
            f.write(f"panel {i}: {n}\n")
