"""Geometric editing method adapters for GeoBench evaluation.

Input contract (per sample):
  - ori_img: original input image (RGB)
  - ori_mask: source object/location mask (white = active region)
  - tgt_mask: desired target geometry/location mask (white = active region)
  - prompt: textual description, typically ``4v_caption`` from GeoBench
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageOps

from eval.data.geobench_geometry import (
    apply_edit_param_to_image,
    verify_tgt_image_mask_adherence,
)
from eval.logging_utils import get_logger
from eval.methods.geo_base import GeometricEditMethod
from eval.utils import to_grayscale, to_rgb

_REPO_ROOT = Path(__file__).resolve().parents[2]

logger = get_logger(__name__)


def _ensure_repo_root() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


class BCDMFluxGeoMethod(GeometricEditMethod):
    """FLUX EditPipeline adapter for GeoBench (socket-style masks, identical prompts)."""

    name = "flux_geom"

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        config_path: str = "backbone/configs/geom_edit_pipeline.yaml",
        model: str = "klein",
        offload: bool = False,
        canvas_size: int = 512,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config_path = config_path
        self.model = model
        self.offload = offload
        self.canvas_size = canvas_size
        self._pipe = None
        self._config: dict | None = None
        self._pt_gen = None

    def load(self) -> None:
        _ensure_repo_root()
        from backbone.edit.geom_controller import EditPipeline
        from util.internlm import TextpromptGen

        cfg_path = _REPO_ROOT / self.config_path
        if not cfg_path.exists():
            raise FileNotFoundError(f"EditPipeline config not found: {cfg_path}")

        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._config = {k: v for k, v in raw.items() if k not in ("geometry", "inputs")}

        self._pipe = EditPipeline(
            offload=self.offload,
            model=self.model,
            device=str(self.device),
        )

        prompt_root = _REPO_ROOT / "eval" / "output" / "geom_prompt_gen"
        prompt_root.mkdir(parents=True, exist_ok=True)
        self._pt_gen = TextpromptGen(str(prompt_root), False)

        self._loaded = True

    def infer(
        self,
        ori_img: Image.Image,
        ori_mask: Image.Image,
        tgt_mask: Image.Image,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        self._ensure_loaded()
        if self._pipe is None or self._config is None:
            raise RuntimeError("EditPipeline is not initialized.")
        if self._pt_gen is None:
            raise RuntimeError("Gemini prompt generator is not initialized.")

        _ensure_repo_root()
        from backbone.edit.socket_edit import masked_source_for_caption
        from eval.methods.geobench_spec import build_geobench_geometry_spec

        size = self.canvas_size
        rgb = self._as_rgb(ori_img).resize((size, size), Image.Resampling.LANCZOS)
        ori_m = self._prepare_geo_mask(ori_mask, size)
        tgt_m = self._prepare_geo_mask(tgt_mask, size)
        ori_m.save("ori_m.png")
        tgt_m.save("tgt_m.png")

        edit_param = kwargs.get("edit_param") or kwargs.get("edit_param_json")
        if edit_param is None:
            raise ValueError(
                "BCDMFluxGeoMethod requires edit_param or edit_param_json in infer kwargs."
            )

        tgt_rgb, warped_object_mask = apply_edit_param_to_image(
            rgb,
            ori_m,
            edit_param,
            size=size,
            background=rgb,
        )
        warped_object_mask.save("warped_object_mask.png")
        tgt_rgb.save("tgt_rgb.png")
        adherence = verify_tgt_image_mask_adherence(warped_object_mask, tgt_m)
        logger.info(
            "edit_param tgt_image mask adherence: warped_mask_tgt_iou=%.4f valid=%s",
            adherence["warped_mask_tgt_iou"],
            adherence["valid"],
        )

        source_mask_for_caption = ori_m

        masked_source = masked_source_for_caption(rgb, source_mask_for_caption)

        source_caption = self._pt_gen.describe_edit_source_without_mask(masked_source)
        logger.info("Source-scene caption generated for inpaint.")

        def describe_composed(composed_image: Image.Image) -> str:
            composed_caption = self._pt_gen.describe_composed_edit_image(composed_image)
            # composed_image.save("composed_image.png")
            logger.info("Composed image description generated for refine pass.")
            return composed_caption

        spec = build_geobench_geometry_spec(ori_m, warped_object_mask, source_caption, size=size)
        output_dir = kwargs.get("debug_dir") or "eval/output"
        result = self._pipe.run(
            src_image=rgb,
            tgt_image=tgt_rgb,
            spec=spec,
            config=self._config,
            output_dir=output_dir,
            composition_prompt_callback=describe_composed,
        )
        return self._as_rgb(result)

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._pt_gen = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()


from eval.methods.design_edit_geom import DesignEditGeoMethod
from eval.methods.freefine_geom import FreeFineGeoMethod

GEO_METHOD_REGISTRY: dict[str, type[GeometricEditMethod]] = {
    "flux_geom": BCDMFluxGeoMethod,
    "bcdm_flux_geom": BCDMFluxGeoMethod,
    "design_edit": DesignEditGeoMethod,
    "freefine": FreeFineGeoMethod,
    "freefine_geom": FreeFineGeoMethod,
}


_FLUX_ONLY_KWARGS = frozenset({"config_path", "model", "canvas_size"})
_OFFLOAD_METHODS = frozenset({"flux_geom", "bcdm_flux_geom", "design_edit"})


def get_geo_method(
    name: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> GeometricEditMethod:
    """Instantiate a registered geometric editing method."""
    if name not in GEO_METHOD_REGISTRY:
        known = ", ".join(sorted(GEO_METHOD_REGISTRY))
        raise ValueError(f"Unknown geometric method '{name}'. Available: {known}")
    cls = GEO_METHOD_REGISTRY[name]
    if name not in ("flux_geom", "bcdm_flux_geom"):
        kwargs = {k: v for k, v in kwargs.items() if k not in _FLUX_ONLY_KWARGS}
    if name not in _OFFLOAD_METHODS:
        kwargs.pop("offload", None)
    return cls(device=device, dtype=dtype, **kwargs)
