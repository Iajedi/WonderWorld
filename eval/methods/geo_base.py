"""Base class for GeoBench geometric editing method adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image, ImageOps

from eval.utils import to_grayscale


class GeometricEditMethod(ABC):
    """Unified interface for geometric image editing pipelines."""

    name: str = "geo_base"

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights / pipeline."""

    @abstractmethod
    def infer(
        self,
        ori_img: Image.Image,
        ori_mask: Image.Image,
        tgt_mask: Image.Image,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        """Run geometric editing and return an RGB PIL image."""

    def unload(self) -> None:
        """Release model resources."""
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(f"Method '{self.name}' is not loaded. Call load() first.")

    @staticmethod
    def _as_rgb(image: Image.Image) -> Image.Image:
        if image.mode == "RGB":
            return image
        return image.convert("RGB")

    @staticmethod
    def _prepare_geo_mask(mask: Image.Image, size: int) -> Image.Image:
        """Resize a GeoBench mask for the edit pipeline."""
        gray = to_grayscale(ImageOps.invert(mask)).resize((size, size), Image.Resampling.NEAREST)
        return gray
