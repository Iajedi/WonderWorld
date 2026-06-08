"""FreeFine adapter for GeoBench geometric editing evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from eval.config import (
    FREEFINE_CANVAS_SIZE,
    FREEFINE_CHECKOUT,
    FREEFINE_MASK_DILATION,
    FREEFINE_SD15_PATH,
)
from eval.data.geobench_geometry import apply_edit_param_to_image, parse_edit_param
from eval.logging_utils import get_logger
from eval.methods._external_import import checkout_sys_path
from eval.methods.geo_base import GeometricEditMethod
from eval.utils import binarize_mask, to_grayscale, to_rgb

logger = get_logger(__name__)


def _ensure_freefine_checkout() -> Path:
    checkout = Path(FREEFINE_CHECKOUT)
    if not checkout.is_dir():
        raise FileNotFoundError(
            f"FreeFine checkout not found at {checkout}. "
            "Run: bash eval/scripts/setup_external_baselines.sh"
        )
    model_py = checkout / "src" / "demo" / "model.py"
    if not model_py.is_file():
        raise FileNotFoundError(
            f"FreeFine model.py missing at {model_py}. Re-run setup_external_baselines.sh"
        )
    return checkout


def _pil_rgb_to_numpy(image: Image.Image, size: int) -> np.ndarray:
    rgb = to_rgb(image).resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(rgb, dtype=np.uint8)


def _pil_mask_to_numpy(mask: Image.Image, size: int, *, dilate: int | None = None) -> np.ndarray:
    gray = binarize_mask(to_grayscale(mask)).resize((size, size), Image.Resampling.NEAREST)
    arr = np.asarray(gray, dtype=np.uint8)
    if dilate is not None and dilate > 0:
        arr = cv2.dilate(arr, np.ones((dilate, dilate), np.uint8), iterations=1)
    return np.stack([arr, arr, arr], axis=-1)


def _mask_to_single_channel(mask_np: np.ndarray) -> np.ndarray:
    if mask_np.ndim == 3:
        return mask_np[:, :, 0]
    return mask_np


def _as_hwc_uint8(image: np.ndarray | torch.Tensor, size: int) -> np.ndarray:
    """Normalize FreeFine pipeline output to HxWx3 uint8 (matches upstream eval scripts)."""
    if isinstance(image, torch.Tensor):
        tensor = image[0] if image.dim() == 4 else image
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        arr = tensor.detach().float().cpu().numpy()
        if arr.max() <= 1.0:
            arr = arr * 255.0
        image = arr
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr[:3], (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[:2] != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return arr[..., :3].astype(np.uint8)


def _blend_inp_background(ori_img: np.ndarray, ori_mask: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """BrushNet-style composite (see freefine_batch_infer_bggen_2d.py)."""
    size = ori_img.shape[0]
    generated = _as_hwc_uint8(generated, size)
    ori_img = _as_hwc_uint8(ori_img, size)
    mask = _mask_to_single_channel(np.asarray(ori_mask)).astype(np.float32)
    mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0
    mask_np = 1.0 - (1.0 - (mask > 0).astype(np.float32)) * (1.0 - mask_blurred)
    if mask_np.ndim == 2:
        mask_np = mask_np[:, :, None]
    blended = ori_img * (1.0 - mask_np) + generated * mask_np
    return blended.astype(np.uint8)


class FreeFineGeoMethod(GeometricEditMethod):
    """FreeFine (SD1.5) adapter for GeoBench 2D geometric edits."""

    name = "freefine"

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        checkout_path: str | Path | None = None,
        pretrained_model_path: str | None = None,
        canvas_size: int = FREEFINE_CANVAS_SIZE,
        mask_dilation: int = FREEFINE_MASK_DILATION,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.checkout_path = Path(checkout_path or FREEFINE_CHECKOUT)
        self.pretrained_model_path = pretrained_model_path or FREEFINE_SD15_PATH
        self.canvas_size = canvas_size
        self.mask_dilation = mask_dilation
        self._model = None
        self._bggen_controller = None
        self._register_bggen_control = None

    def load(self) -> None:
        checkout = _ensure_freefine_checkout()

        with checkout_sys_path(checkout):
            from diffusers import DDIMScheduler
            from src.demo.model import FreeFinePipeline
            from src.utils.attention import (
                Attention_Modulator,
                register_attention_control,
                register_attention_control_4bggen,
            )

            logger.info(
                "Loading FreeFine from %s (model=%s, device=%s)",
                checkout,
                self.pretrained_model_path,
                self.device,
            )
            self._model = FreeFinePipeline.from_pretrained(
                self.pretrained_model_path,
                torch_dtype=torch.float32,
            ).to(self.device)
            self._model._progress_bar_config = {"disable": True}
            self._model.scheduler = DDIMScheduler.from_config(self._model.scheduler.config)

            edit_controller = Attention_Modulator(start_layer=10)
            self._model.controller = edit_controller
            register_attention_control(self._model, edit_controller)
            self._model.modify_unet_forward()
            self._model.enable_attention_slicing()
            try:
                self._model.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                logger.warning("xformers not available for FreeFine: %s", exc)

            bggen_controller = Attention_Modulator(start_layer=10)
            self._bggen_controller = bggen_controller
            self._register_bggen_control = register_attention_control_4bggen
        self._loaded = True

    def _load_or_generate_background(
        self,
        *,
        sample_id: str | None,
        cache_dir: Path | None,
        ori_img: np.ndarray,
        dilated_mask: np.ndarray,
    ) -> np.ndarray:
        if sample_id and cache_dir is not None:
            cache_path = cache_dir / f"{sample_id}.png"
            if cache_path.is_file() and cache_path.stat().st_size > 0:
                cached = cv2.imread(str(cache_path))
                if cached is not None:
                    cached = cv2.cvtColor(cached, cv2.COLOR_BGR2RGB)
                    if cached.shape[:2] == ori_img.shape[:2]:
                        return cached

        assert self._model is not None
        assert self._bggen_controller is not None
        assert self._register_bggen_control is not None

        self._model.controller = self._bggen_controller
        self._register_bggen_control(self._model, self._bggen_controller)

        bggen_params = {
            "ori_img": ori_img,
            "ori_mask": dilated_mask,
            "guidance_text": "empty scene",
            "guidance_scale": 7.5,
            "eta": 1.0,
            "end_scale": 0.5,
            "end_step": 35,
            "num_step": 50,
            "start_step": 1,
            "seed": 42,
            "return_intermediates": False,
        }
        generated = self._model.FreeFine_background_generation(**bggen_params)
        inp_bg = _blend_inp_background(ori_img, dilated_mask, generated)

        if sample_id and cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{sample_id}.png"
            cv2.imwrite(str(cache_path), cv2.cvtColor(inp_bg, cv2.COLOR_RGB2BGR))

        return inp_bg

    def infer(
        self,
        ori_img: Image.Image,
        ori_mask: Image.Image,
        tgt_mask: Image.Image,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        del prompt
        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError("FreeFine model is not initialized.")

        edit_param = kwargs.get("edit_param") or kwargs.get("edit_param_json")
        if edit_param is None:
            raise ValueError("FreeFineGeoMethod requires edit_param_json in infer kwargs.")

        size = self.canvas_size
        ori_mask = ImageOps.invert(ori_mask)
        ori_np = _pil_rgb_to_numpy(ori_img, size)
        ori_mask_np = _pil_mask_to_numpy(ori_mask, size)
        dilated_mask_np = _pil_mask_to_numpy(ori_mask, size, dilate=self.mask_dilation)

        cache_dir = kwargs.get("cache_dir")
        if cache_dir is not None and not isinstance(cache_dir, Path):
            cache_dir = Path(cache_dir)
        sample_id = kwargs.get("sample_id")
        obj_label = str(kwargs.get("obj_label") or "")

        inp_background = self._load_or_generate_background(
            sample_id=sample_id,
            cache_dir=cache_dir,
            ori_img=ori_np,
            dilated_mask=dilated_mask_np,
        )

        coarse_pil, warped_mask_pil = apply_edit_param_to_image(
            ori_img,
            ori_mask,
            edit_param,
            size=size,
            background=Image.fromarray(inp_background, mode="RGB"),
        )
        coarse_np = np.asarray(coarse_pil, dtype=np.uint8)
        target_mask_np = np.asarray(warped_mask_pil, dtype=np.uint8)
        target_mask_np = np.stack([target_mask_np, target_mask_np, target_mask_np], axis=-1)

        from src.utils.attention import Attention_Modulator, register_attention_control

        checkout = _ensure_freefine_checkout()
        with checkout_sys_path(checkout):
            edit_controller = Attention_Modulator(start_layer=10)
            self._model.controller = edit_controller
            register_attention_control(self._model, edit_controller)

            gen_params = {
                "ori_img": ori_np,
                "ori_mask": ori_mask_np,
                "coarse_input": coarse_np,
                "target_mask": target_mask_np,
                "guidance_text": obj_label,
                "guidance_scale": 7.5,
                "eta": 1.0,
                "end_scale": 0.0,
                "end_step": 50,
                "num_step": 50,
                "start_step": 35,
                "seed": 42,
                "draw_mask": None,
                "return_intermediates": False,
                "use_auto_draw": True,
                "reduce_inp_artifacts": True,
                "cons_area": target_mask_np,
            }
            result = self._model.FreeFine_generation(**gen_params)
        if isinstance(result, tuple):
            result = result[0]
        result = _as_hwc_uint8(result, size)
        return Image.fromarray(result, mode="RGB")

    def unload(self) -> None:
        self._model = None
        self._bggen_controller = None
        self._register_bggen_control = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()
