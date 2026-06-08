"""Abstract base class for inpainting methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image


class InpaintingMethod(ABC):
    """Unified interface for text-guided inpainting pipelines."""

    name: str = "base"

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
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        """Run inpainting and return an RGB PIL image."""

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
