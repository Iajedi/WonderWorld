"""Inpainting method adapters."""

from __future__ import annotations

import torch

from eval.methods.base import InpaintingMethod
from eval.methods.bcdm_flux_klein import BCDMFluxKleinMethod
from eval.methods.flux_fill_dev import FluxFillDevMethod
from eval.methods.sd15_inpaint import SD15InpaintMethod
from eval.methods.sd2_inpaint import SD2InpaintMethod

METHOD_REGISTRY: dict[str, type[InpaintingMethod]] = {
    "sd15_inpaint": SD15InpaintMethod,
    "sd15": SD15InpaintMethod,
    "sd2_inpaint": SD2InpaintMethod,
    "sd2": SD2InpaintMethod,
    "flux_fill_dev": FluxFillDevMethod,
    "flux1_fill": FluxFillDevMethod,
    "bcdm_flux_klein": BCDMFluxKleinMethod,
    "flux_klein": BCDMFluxKleinMethod,
}


def get_method(
    name: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> InpaintingMethod:
    """Instantiate a registered inpainting method."""
    if name not in METHOD_REGISTRY:
        known = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unknown method '{name}'. Available: {known}")
    return METHOD_REGISTRY[name](device=device, dtype=dtype)
