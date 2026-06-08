"""Shared helpers for Hugging Face diffusers method adapters."""

from __future__ import annotations

import torch
from PIL import Image

from eval.utils import to_grayscale, to_rgb

DEFAULT_SIZE = 512


def prepare_inpaint_inputs(
    image: Image.Image,
    mask: Image.Image,
    *,
    size: int = DEFAULT_SIZE,
) -> tuple[Image.Image, Image.Image]:
    """Resize image/mask to a common square size in eval mask convention."""
    rgb = to_rgb(image)
    mask_l = to_grayscale(mask)
    if rgb.size != (size, size):
        rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)
        mask_l = mask_l.resize((size, size), Image.Resampling.NEAREST)
    return rgb, mask_l


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a deterministic torch Generator on the requested device."""
    gen_device = device.type if device.type in {"cuda", "cpu"} else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator
