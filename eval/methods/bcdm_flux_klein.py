"""BCDM FLUX.2 Klein inpainting adapter for the eval suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from eval.methods.base import InpaintingMethod
from eval.utils import to_grayscale

_REPO_ROOT = Path(__file__).resolve().parents[2]


class BCDMFluxKleinMethod(InpaintingMethod):
    """Wraps backbone BCDMPipeline (FLUX.2 Klein 4B)."""

    name = "bcdm_flux_klein"

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        config_path: str = "backbone/configs/inpaint2.yaml",
        model: str = "klein",
        offload: bool = False,
        seed: int = 42,
        blackout_unknown: bool = True,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config_path = config_path
        self.model = model
        self.offload = offload
        self.seed = seed
        self.blackout_unknown = blackout_unknown
        self._pipe = None
        self._config: dict | None = None

    def load(self) -> None:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))

        from backbone.pipeline import BackbonePipeline

        cfg_path = _REPO_ROOT / self.config_path
        if not cfg_path.exists():
            raise FileNotFoundError(f"BCDM config not found: {cfg_path}")

        with open(cfg_path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        self._pipe = BackbonePipeline(
            offload=self.offload,
            device=str(self.device),
        )
        self._loaded = True

    def infer(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        self._ensure_loaded()
        if self._pipe is None or self._config is None:
            raise RuntimeError("BCDM pipeline is not initialized.")

        prompt_src = str(kwargs.get("prompt_src", prompt))
        prompt_tgt = str(kwargs.get("prompt_tgt", prompt))

        rgb = self._as_rgb(image)
        mask_l = to_grayscale(mask).resize(rgb.size, Image.Resampling.NEAREST)
        if rgb.size != (512, 512):
            rgb = rgb.resize((512, 512), Image.Resampling.LANCZOS)
            mask_l = mask_l.resize((512, 512), Image.Resampling.NEAREST)

        mask_np = (
            np.array(mask_l, dtype=np.float32) / 255.0
        ).reshape(1, 1, 512, 512)

        result = self._pipe.run(
            image=rgb,
            mask=mask_np,
            prompt_src=prompt_src,
            prompt_tgt=prompt_tgt,
            config=self._config,
            output_dir=kwargs.get("output_dir"),
            blackout_unknown=self.blackout_unknown,
        )
        return self._as_rgb(result)

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()
