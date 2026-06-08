"""Geometric editing on top of :class:`~edit.controller.BCDMPipeline`."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageOps

try:
    from .controller import BCDMPipeline
    from ..geometry.spec import EditType, GeometrySpec
    from ..geometry.utils import (
        build_boundary_blur_mask,
        build_inpainting_mask,
        geometry_mask_to_controller_np,
        inverse_affine_coeffs_for_pil,
        mask_tensor_to_pil_l,
    )
except ImportError:
    from backbone.edit.controller import BCDMPipeline
    from backbone.geometry.spec import EditType, GeometrySpec
    from backbone.geometry.utils import (
        build_boundary_blur_mask,
        build_inpainting_mask,
        geometry_mask_to_controller_np,
        inverse_affine_coeffs_for_pil,
        mask_tensor_to_pil_l,
    )

_CANVAS_HW: Tuple[int, int] = (512, 512)


def _ensure_rgb_size(im: Image.Image, size_hw: Tuple[int, int]) -> Image.Image:
    h, w = size_hw
    return im.convert("RGB").resize((w, h), Image.LANCZOS)


class EditPipeline(BCDMPipeline):
    """Two-stage route: inpaint source, compose with (inverse-warped) target, refine seams."""

    def __init__(
        self,
        offload: bool = False,
        model: str = "klein",
        device: str = "cuda",
        base_pipeline: Optional[BCDMPipeline] = None,
    ):
        if base_pipeline is None:
            super().__init__(offload, model, device=device)
        else:
            self.wrapper = base_pipeline.wrapper
            self.model_type = base_pipeline.model_type
            self.device = base_pipeline.device
            self.offload = base_pipeline.offload

    @torch.no_grad()
    def run(
        self,
        src_image: Image.Image,
        tgt_image: Optional[Image.Image],
        spec: GeometrySpec,
        config: Dict[str, Any],
        output_dir: str = "outputs/geom",
        composition_prompt_callback: Optional[Callable[[Image.Image], Optional[str]]] = None,
    ) -> Image.Image:
        if spec.edit_type == EditType.COMPOSE_MULTI:
            if tgt_image is None:
                raise ValueError("Composition task requires a target image")
        else:
            tgt_image = src_image

        assert tgt_image is not None
        if spec.mask_tgt is None:
            raise ValueError("Geometry edit requires mask_tgt for composition / seam mask")

        src_image = _ensure_rgb_size(src_image, _CANVAS_HW)
        tgt_image = _ensure_rgb_size(tgt_image, _CANVAS_HW)

        if spec.edit_type == EditType.COPY:
            inpainted_image = src_image
        else:
            inpaint_mask = build_inpainting_mask(spec)
            if bool(torch.any(inpaint_mask.detach().float() > 0.5)):
                inpaint_mask_np = geometry_mask_to_controller_np(inpaint_mask, _CANVAS_HW)
                inpainted_image = super().run(
                    src_image,
                    mask=inpaint_mask_np,
                    prompt_src=spec.prompt_inpaint,
                    prompt_tgt=spec.prompt_inpaint,
                    config=config,
                    output_dir=None,
                )
            else:
                inpainted_image = src_image

        if spec.edit_type == EditType.COMPOSE_MULTI and not spec.compose_layers:
            raise ValueError("COMPOSE_MULTI requires non-empty compose_layers (use GeometrySpec.for_compose_multi)")

        if spec.edit_type == EditType.COMPOSE_MULTI and spec.compose_layers:
            composed_image = inpainted_image
            for T_i, paste_m in spec.compose_layers:
                coeffs = inverse_affine_coeffs_for_pil(T_i)
                warped = tgt_image.transform(
                    tgt_image.size,
                    Image.AFFINE,
                    coeffs,
                    resample=Image.BILINEAR,
                )
                paste_pil = mask_tensor_to_pil_l(paste_m, _CANVAS_HW)
                composite_mask = ImageOps.invert(paste_pil)
                composed_image = Image.composite(composed_image, warped, composite_mask).convert("RGB")
        else:
            if spec.transform_matrix is not None:
                coeffs = inverse_affine_coeffs_for_pil(spec.transform_matrix)
                tgt_image = tgt_image.transform(
                    tgt_image.size,
                    Image.AFFINE,
                    coeffs,
                    resample=Image.BILINEAR,
                )

            mask_composite_pil = mask_tensor_to_pil_l(spec.mask_tgt, _CANVAS_HW)
            composite_mask = ImageOps.invert(mask_composite_pil)
            composed_image = Image.composite(inpainted_image, tgt_image, composite_mask).convert("RGB")

        if output_dir is not None and bool(config.get("debug", False)):
            os.makedirs(output_dir, exist_ok=True)
            composed_image.save(os.path.join(output_dir, "composed_image.png"))

        band_k = int(config.get("seam_band_kernel", 7))
        blur_r = float(config.get("seam_blur_radius", 2.0))
        sigma_in = float(config.get("seam_sigma_inside", 2.5))
        sigma_out = float(config.get("seam_sigma_outside", 12.0))
        refine_mask_np = build_boundary_blur_mask(
            spec.mask_tgt,
            size_hw=_CANVAS_HW,
            band_kernel_size=band_k,
            gaussian_radius=blur_r,
            sigma_inside=sigma_in,
            sigma_outside=sigma_out,
        )
        target_mask_np = geometry_mask_to_controller_np(spec.mask_tgt, _CANVAS_HW)
        late_refine_mask_np = np.maximum(refine_mask_np, target_mask_np).astype(np.float32)
        late_edit_steps = int(config.get("L", 0))

        # Use a tighter reinjection mask in the seam-refinement pass so the
        # pasted object structure stays anchored while its appearance harmonizes.
        refine_config = dict(config)
        refine_config["reinject_unknown_expand_px"] = 4
        refine_config["reinject_mask_dilate"] = 1

        # refine_mask_pil = Image.fromarray((refine_mask_np.squeeze() * 255).astype(np.uint8), mode="L")
        # refine_mask_pil.save("refine_mask.png")

        prompt_refine = spec.prompt_refine if spec.prompt_refine is not None else spec.prompt_inpaint
        if composition_prompt_callback is not None:
            callback_prompt = composition_prompt_callback(composed_image)
            if callback_prompt:
                prompt_refine = callback_prompt
        return super().run(
            composed_image,
            mask=refine_mask_np,
            prompt_src=spec.prompt_inpaint,
            prompt_tgt=prompt_refine,
            config=refine_config,
            output_dir=output_dir,
            blackout_unknown=False,
            late_edit_mask=late_refine_mask_np,
            late_edit_steps=late_edit_steps,
        )
