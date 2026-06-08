"""DesignEdit adapter for GeoBench geometric editing evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from eval.config import DESIGNEDIT_CANVAS_SIZE, DESIGNEDIT_CHECKOUT, DESIGNEDIT_SDXL_PATH
from eval.data.geobench_geometry import parse_edit_param
from eval.logging_utils import get_logger
from eval.methods._external_import import checkout_sys_path
from eval.methods.geo_base import GeometricEditMethod
from eval.utils import binarize_mask, to_grayscale, to_rgb

logger = get_logger(__name__)


def _ensure_designedit_checkout() -> Path:
    checkout = Path(DESIGNEDIT_CHECKOUT)
    if not checkout.is_dir():
        raise FileNotFoundError(
            f"DesignEdit checkout not found at {checkout}. "
            "Run: bash eval/scripts/setup_external_baselines.sh"
        )
    model_py = checkout / "src" / "demo" / "model.py"
    if not model_py.is_file():
        raise FileNotFoundError(
            f"DesignEdit model.py missing at {model_py}. Re-run setup_external_baselines.sh"
        )
    return checkout


class DesignEditGeoMethod(GeometricEditMethod):
    """DesignEdit (SDXL) adapter for GeoBench 2D geometric edits."""

    name = "design_edit"

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        checkout_path: str | Path | None = None,
        pretrained_model_path: str | None = None,
        canvas_size: int = DESIGNEDIT_CANVAS_SIZE,
        offload: bool = False,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.checkout_path = Path(checkout_path or DESIGNEDIT_CHECKOUT)
        self.pretrained_model_path = pretrained_model_path or DESIGNEDIT_SDXL_PATH
        self.canvas_size = canvas_size
        self.offload = offload
        self._model = None

    def load(self) -> None:
        checkout = _ensure_designedit_checkout()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with checkout_sys_path(checkout):
            from src.demo.model import DesignEdit

            device_id = (
                self.device.index
                if self.device.type == "cuda" and self.device.index is not None
                else 0
            )
            logger.info(
                "Loading DesignEdit from %s (model=%s, device=%s, offload=%s)",
                checkout,
                self.pretrained_model_path,
                self.device,
                self.offload,
            )
            self._model = DesignEdit(
                pretrained_model_path=self.pretrained_model_path,
                device=device_id if self.device.type == "cuda" else None,
            )
            pipe = self._model.ldm_model
            if self.offload and self.device.type == "cuda":
                pipe.to("cpu")
                torch.cuda.empty_cache()
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload(gpu_id=device_id)
                elif hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload(gpu_id=device_id)
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        self._loaded = True

    def infer(
        self,
        ori_img: Image.Image,
        ori_mask: Image.Image,
        tgt_mask: Image.Image,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        del tgt_mask, prompt  # DesignEdit 2D geo edit uses edit_param only.
        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError("DesignEdit model is not initialized.")

        edit_param = kwargs.get("edit_param") or kwargs.get("edit_param_json")
        if edit_param is None:
            raise ValueError("DesignEditGeoMethod requires edit_param_json in infer kwargs.")

        ori_rgb = np.asarray(to_rgb(ori_img), dtype=np.uint8)
        img_mask = ImageOps.invert(ori_mask)
        img_mask.save("img_mask.png")
        gray_mask = np.asarray(binarize_mask(to_grayscale(img_mask)), dtype=np.uint8)
        ori_m = np.stack([gray_mask, gray_mask, gray_mask], axis=-1)

        dx, dy, _, _, _, rz, sx, _, _ = parse_edit_param(edit_param)
        dx /= 512
        dy /= -512
        rz = -rz

        with torch.no_grad():
            generated_results = self._model.infer_2d_edit(
                ori_rgb,
                ori_rgb,
                ori_m,
                dx,
                dy,
                sx,
                rz,
            )

        out = generated_results[0]
        size = self.canvas_size
        if out.shape[:2] != (size, size):
            out = cv2.resize(out, (size, size), interpolation=cv2.INTER_LANCZOS4)
        return Image.fromarray(out.astype(np.uint8), mode="RGB")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()
