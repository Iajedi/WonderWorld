from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter
from kornia.morphology import dilation, erosion

try:
    from .spec import EditType, GeometrySpec
except ImportError:
    from backbone.geometry.spec import EditType, GeometrySpec

try:
    from scipy.ndimage import distance_transform_edt as _scipy_edt
except ImportError:  # pragma: no cover
    _scipy_edt = None


# Union user mask with inpainting mask (if we need structural completion like FreeFine)
def build_inpainting_mask(spec: GeometrySpec) -> torch.Tensor:
    # We union the user mask with the mask_src if it exists
    if spec.mask_user is not None:
        return torch.logical_or(spec.mask_src, spec.mask_user)
    else:
        return spec.mask_src

# Convert geometry mask to numpy array for our backbone for geometric editing
def geom_mask_to_np(mask: torch.Tensor, size_hw: Tuple[int, int] = (512, 512)) -> np.ndarray:
    m = mask.detach().float().cpu().numpy()
    while m.ndim > 2: # Reduction of leading dimensions
        m = m.squeeze(0)
    h, w = size_hw
    # Resize and convert to binarised mask
    pil_m = Image.fromarray((m > 0.5).astype(np.uint8) * 255, mode="L").resize((w, h), Image.NEAREST)
    return (np.asarray(pil_m, dtype=np.float32) / 255.0).reshape(1, 1, h, w)


# Convert binary mask tensor to PIL image for our backbone for composition
def mask_tensor_to_pil_l(mask: torch.Tensor, size_hw: Tuple[int, int]) -> Image.Image:
    m = mask.detach().float().cpu().numpy()
    while m.ndim > 2: # Reduction of leading dimensions
        m = m.squeeze(0)
    h, w = size_hw # Output size
    return Image.fromarray((m > 0.5).astype(np.uint8) * 255, mode="L").resize((w, h), Image.NEAREST)


def inverse_affine_coeffs_for_pil(transform: torch.Tensor):
    # Six coefficients for for the inverse of a 3x3 homogeneous matrix.
    inv = torch.linalg.inv(transform.to(dtype=torch.float64)) # Float 64 for stability
    return ( # 6 dofs
        float(inv[0, 0]),
        float(inv[0, 1]),
        float(inv[0, 2]),
        float(inv[1, 0]),
        float(inv[1, 1]),
        float(inv[1, 2]),
    )

# Build boundary blur mask (seam mask M_{seam}) to blend the object with the background
def build_boundary_blur_mask(
    mask: torch.Tensor,
    size_hw: Tuple[int, int] = (512, 512),
    gaussian_radius: float = 5.0,
):
    h, w = size_hw
    mask = mask.unsqueeze(0) # Add batch dimension
    # Binarise mask
    fg = (mask > 0.5).to(dtype=torch.float32)

    # Dilate and erode the mask to get the boundary
    dilated_mask = dilation(fg, kernel=torch.ones(3, 3))
    eroded_mask = erosion(fg, kernel=torch.ones(3, 3))

    # We get the band by subtracting the eroded mask from the dilated mask
    band = dilated_mask - eroded_mask
    # Apply a Gaussian blur to the band
    band_img = Image.fromarray(band.numpy()[0, 0, :, :].astype(np.uint8) * 255, mode="L")
    band_img = band_img.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
    band = np.asarray(band_img, dtype=np.float32) / 255.0
    return band.reshape(1, 1, h, w)
