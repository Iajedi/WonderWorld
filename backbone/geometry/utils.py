from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter

from geometry.spec import EditType, GeometrySpec

try:
    from scipy.ndimage import distance_transform_edt as _scipy_edt
except ImportError:  # pragma: no cover
    _scipy_edt = None


def requires_inpainting(spec: GeometrySpec) -> bool:
    """Returns True if the geometry spec requires first-stage inpainting."""
    return spec.edit_type in [EditType.MOVE, EditType.RESIZE, EditType.COMPOSE_MULTI] or spec.mask_user is not None

def build_inpainting_mask(spec: GeometrySpec) -> torch.Tensor:
    """Builds the inpainting mask for the geometry spec."""
    assert requires_inpainting(spec), "Geometry spec does not require inpainting"
    # We union the user mask with the mask_src if it exists
    if spec.mask_user is not None:
        return torch.logical_or(spec.mask_src, spec.mask_user)
    else:
        return spec.mask_src


def geometry_mask_to_controller_np(mask: torch.Tensor, size_hw: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Float mask ``[1, 1, H, W]`` in ``[0, 1]`` for :class:`~edit.controller.BCOTHVEPipeline`."""
    m = mask.detach().float().cpu().numpy()
    while m.ndim > 2:
        m = m.squeeze(0)
    if m.ndim != 2:
        raise ValueError(f"mask must reduce to HxW, got {mask.shape}")
    h, w = size_hw
    pil_m = Image.fromarray((m > 0.5).astype(np.uint8) * 255, mode="L").resize((w, h), Image.NEAREST)
    return (np.asarray(pil_m, dtype=np.float32) / 255.0).reshape(1, 1, h, w)


def mask_tensor_to_pil_l(mask: torch.Tensor, size_hw: Tuple[int, int]) -> Image.Image:
    """Binary ``mask`` to mode ``L`` PIL, resized to ``(height, width)`` = ``size_hw``."""
    m = mask.detach().float().cpu().numpy()
    while m.ndim > 2:
        m = m.squeeze(0)
    if m.ndim != 2:
        raise ValueError(f"mask must reduce to HxW, got {mask.shape}")
    h, w = size_hw
    return Image.fromarray((m > 0.5).astype(np.uint8) * 255, mode="L").resize((w, h), Image.NEAREST)


def inverse_affine_coeffs_for_pil(transform_3x3: torch.Tensor) -> tuple[float, float, float, float, float, float]:
    """Six coefficients for ``Image.AFFINE`` from a **forward** 3×3 homogeneous matrix.

    ``GeometrySpec.transform_matrix`` maps pixel coordinates in the sense
    ``[x'; y'; 1] = T @ [x; y; 1]`` (column vectors).  Pillow's affine resampling
    expects the **inverse** map from *output* pixel *(x, y)* to *input* sample
    position ``(x', y')``::

        x' = a*x + b*y + c
        y' = d*x + e*y + f

    so the returned tuple is ``(a,b,c,d,e,f)`` built from ``T^{-1}``.
    """
    t = transform_3x3.to(dtype=torch.float64)
    if t.shape != (3, 3):
        raise ValueError(f"transform_matrix must be 3x3, got {tuple(t.shape)}")
    inv = torch.linalg.inv(t)
    return (
        float(inv[0, 0]),
        float(inv[0, 1]),
        float(inv[0, 2]),
        float(inv[1, 0]),
        float(inv[1, 1]),
        float(inv[1, 2]),
    )


def _edt_to_nearest_zero(arr01: np.ndarray) -> np.ndarray:
    """Euclidean distance to the nearest pixel with value 0 in ``arr01`` (uint8 0/1).

    SciPy is used when available; otherwise a 4-connected integer BFS approximation.
    """
    a = (arr01 != 0).astype(np.uint8)
    if _scipy_edt is not None:
        return _scipy_edt(a).astype(np.float32)
    h, w = a.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    q: deque[tuple[int, int]] = deque()
    for y in range(h):
        for x in range(w):
            if a[y, x] == 0:
                dist[y, x] = 0.0
                q.append((y, x))
    while q:
        y, x = q.popleft()
        d0 = dist[y, x]
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= h or nx < 0 or nx >= w or a[ny, nx] != 1:
                continue
            nd = d0 + 1.0
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                q.append((ny, nx))
    dist[a == 0] = 0.0
    dist[~np.isfinite(dist)] = 0.0
    return dist


def build_boundary_blur_mask(
    mask: torch.Tensor,
    size_hw: Tuple[int, int] = (512, 512),
    band_kernel_size: int = 7,
    gaussian_radius: float = 5.0,
    sigma_inside: float = 2.5,
    sigma_outside: float = 12.0,
) -> np.ndarray:
    """Soft seam mask with *asymmetric* falloff around the foreground boundary.

    Weights are high on the paste boundary and drop **quickly** when moving
    **into** the masked (foreground) region—so the interior is largely
    preserved—and **slowly** when moving **outward** into the background for a
    gentle taper. A light Gaussian is applied at the end for continuity.

    Parameters
    ----------
    mask
        Tensor reducing to ``[H, W]`` (values > 0.5 treated as foreground).
    size_hw
        Output spatial size ``(height, width)``.
    band_kernel_size
        Odd side length for max/min filters (morphological dilate / erode).
    gaussian_radius
        Final :class:`PIL.ImageFilter.GaussianBlur` radius (anti-alias).
    sigma_inside
        Distance scale (pixels) for the **inward** (foreground) falloff; smaller
        ⇒ steeper decay into the mask interior.
    sigma_outside
        Distance scale (pixels) for the **outward** (background) taper; larger
        ⇒ more gradual decay outside the mask.

    Returns
    -------
    numpy.ndarray
        Shape ``[1, 1, H, W]``, float32 in ``[0, 1]`` (1 = strongest seam weight).
    """
    h, w = size_hw
    pil_m = mask_tensor_to_pil_l(mask, size_hw)
    M = (np.asarray(pil_m, dtype=np.float32) / 255.0).clip(0.0, 1.0)
    fg = (M > 0.5).astype(np.uint8)
    # Distance to nearest background (0 in ``fg``) — grows into the interior of the mask.
    d_in = _edt_to_nearest_zero(fg)
    # Distance to nearest foreground (0 in ``1-fg``) — grows into the exterior.
    d_out = _edt_to_nearest_zero(1 - fg)

    s_in = max(float(sigma_inside), 1e-3)
    s_out = max(float(sigma_outside), 1e-3)
    asym = np.exp(-((d_in / s_in) ** 2)) * np.exp(-((d_out / s_out) ** 2))

    k = max(3, int(band_kernel_size) | 1)
    dilated = pil_m.filter(ImageFilter.MaxFilter(k))
    eroded = pil_m.filter(ImageFilter.MinFilter(k))
    band = ImageChops.subtract(dilated, eroded)
    band_f = np.asarray(band, dtype=np.float32) / 255.0

    out = asym * band_f
    out = out / (out.max() + 1e-8)
    out_img = Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    out_img = out_img.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
    a = np.asarray(out_img, dtype=np.float32) / 255.0
    a = np.clip(a, 0.0, 1.0)
    return a.reshape(1, 1, h, w)
