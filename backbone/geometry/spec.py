"""Tensor-centric geometry specification — single source of truth for geometric edits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class EditType(str, Enum):
    MOVE = "move"
    COPY = "copy"
    RESIZE = "resize"
    COMPOSE_MULTI = "compose_multi"


def _squeeze_mask_to_1hw(m: torch.Tensor) -> torch.Tensor:
    t = m
    while t.ndim > 3:
        t = t.squeeze(0)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim != 3 or t.shape[0] != 1:
        raise ValueError(f"mask must be [1, H, W] after squeeze, got {tuple(m.shape)}")
    return t


def affine_transform_mask(
    mask: torch.Tensor,
    transform_matrix: torch.Tensor,
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """Apply an affine geometry to a binary/soft mask via inverse sampling.

    ``mask`` is ``[1, H, W]``.  ``transform_matrix`` is a forward 3×3 map in pixel
    coordinates with column vectors ``[j; i; 1]`` (column ``j``, row ``i``)::

        [j'; i'; 1] = T @ [j; i; 1].

    The returned mask has the same ``(H, W)`` grid as ``mask``.  Each output pixel
    ``(j', i')`` samples ``mask`` at ``T^{-1} (j', i')`` (zero padding outside the
    image), so the support of the mask moves/resizes consistently with
    ``inverse_affine_coeffs_for_pil(T)`` warping of images.
    """
    m = _squeeze_mask_to_1hw(mask)
    _, H, W = m.shape
    device = m.device
    T = transform_matrix.to(device=device, dtype=torch.float64)
    if tuple(T.shape) != (3, 3):
        raise ValueError(f"transform_matrix must be 3×3, got {tuple(T.shape)}")
    inv = torch.linalg.inv(T)

    j_out = torch.arange(W, device=device, dtype=torch.float64).view(1, -1).expand(H, W)
    i_out = torch.arange(H, device=device, dtype=torch.float64).view(-1, 1).expand(H, W)
    ones = torch.ones((H, W), device=device, dtype=torch.float64)
    hom = torch.stack((j_out, i_out, ones), dim=0).reshape(3, H * W)
    src = inv @ hom
    w = src[2].clamp(min=1e-12)
    j_src = (src[0] / w).view(H, W)
    i_src = (src[1] / w).view(H, W)

    grid_x = 2.0 * (j_src + 0.5) / float(W) - 1.0
    grid_y = 2.0 * (i_src + 0.5) / float(H) - 1.0
    # ``grid_sample`` requires the grid dtype to match the sampled tensor (float32 here).
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()

    sampled = F.grid_sample(
        m.float().unsqueeze(0),
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )
    return sampled.squeeze(0).to(dtype=m.dtype)


@dataclass
class GeometrySpec:
    """Full geometry edit specification (masks + prompts + metadata)."""

    # Required first (dataclass field ordering).
    mask_src: torch.Tensor
    prompt_inpaint: str

    mask_tgt: Optional[torch.Tensor] = None
    mask_user: Optional[torch.Tensor] = None
    prompt_refine: Optional[str] = None

    # Forward 3×3 homogeneous pixel map: [x'; y'; 1] = T @ [x; y; 1].  When warping
    # images with PIL, use the **inverse** (PIL.Image.AFFINE expects output→input).
    transform_matrix: Optional[torch.Tensor] = None
    # ``COMPOSE_MULTI``: list of ``(forward_3x3, paste_mask_canvas [1,H,W])`` after
    # ``affine_transform_mask`` (sequential paste from inverse-warped ``tgt``).
    compose_layers: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    edit_type: EditType = EditType.MOVE

    # source_mask: torch.Tensor
    # target_mask: Optional[torch.Tensor] = None
    # source_bbox: Optional[Tuple[int, int, int, int]] = None
    # target_bbox: Optional[Tuple[int, int, int, int]] = None
    # transform_matrix: Optional[torch.Tensor] = None
    # preserve_mask: Optional[torch.Tensor] = None
    # hole_mask: Optional[torch.Tensor] = None
    # object_mask_src: Optional[torch.Tensor] = None
    # object_mask_tgt: Optional[torch.Tensor] = None
    # composition_mask: Optional[torch.Tensor] = None
    # prompt_main: str = ""
    # prompt_inpaint: Optional[str] = None
    # prompt_compose: Optional[str] = None
    # metadata: Dict[str, Any] = field(default_factory=dict)

    # Builders for GeometrySpec
    @staticmethod
    def for_move(mask_src: torch.Tensor, dx: float, dy: float, prompt_inpaint: str, prompt_refine: str, mask_user: Optional[torch.Tensor] = None) -> GeometrySpec:
        m = _squeeze_mask_to_1hw(mask_src)
        device, dtype = m.device, m.dtype
        transform_matrix = torch.tensor(
            [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        mask_tgt = affine_transform_mask(m, transform_matrix)
        return GeometrySpec(
            edit_type=EditType.MOVE,
            mask_src=m,
            mask_tgt=mask_tgt,
            transform_matrix=transform_matrix,
            compose_layers=None,
            prompt_inpaint=prompt_inpaint,
            prompt_refine=prompt_refine,
            mask_user=mask_user,
        )

    @staticmethod
    def for_copy(mask_src: torch.Tensor, dx: float, dy: float, prompt_inpaint: str, prompt_refine: str, mask_user: Optional[torch.Tensor] = None) -> GeometrySpec:
        # Pre: mask has shape [1, H, W]
        # Copy is move but without overriding the target mask.
        spec = GeometrySpec.for_move(mask_src, dx, dy, prompt_inpaint, prompt_refine, mask_user)
        spec.edit_type = EditType.COPY
        return spec

    @staticmethod
    def for_resize(mask_src: torch.Tensor, scale: float, prompt_inpaint: str, prompt_refine: str, mask_user: Optional[torch.Tensor] = None) -> GeometrySpec:
        """Uniform scale about the **foreground centroid** of ``mask_src`` (not top-left).

        Forward map (column ``x`` = ``j``, row ``y`` = ``i``): ``p' = s p + (1-s) c`` with
        ``c = (cx, cy)`` the mean ``(x, y)`` of pixels where ``mask_src > 0.5``.  If the mask
        is empty, ``c`` defaults to the image centre.  ``mask_tgt`` is built by sampling
        ``mask_src`` under the inverse map so it stays aligned with ``transform_matrix``.
        """
        m = _squeeze_mask_to_1hw(mask_src)
        _, H, W = m.shape
        s = float(scale)
        if s <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")

        fg = m[0] > 0.5
        ys, xs = torch.where(fg)
        if ys.numel() == 0:
            cx = torch.tensor((W - 1) * 0.5, device=m.device, dtype=torch.float32)
            cy = torch.tensor((H - 1) * 0.5, device=m.device, dtype=torch.float32)
        else:
            cx = xs.float().mean()
            cy = ys.float().mean()

        tx = cx * (1.0 - s)
        ty = cy * (1.0 - s)
        transform_matrix = torch.tensor(
            [[s, 0.0, tx], [0.0, s, ty], [0.0, 0.0, 1.0]],
            device=m.device,
            dtype=m.dtype,
        )
        mask_tgt = affine_transform_mask(m, transform_matrix)

        return GeometrySpec(
            edit_type=EditType.RESIZE,
            mask_src=m,
            mask_tgt=mask_tgt,
            transform_matrix=transform_matrix,
            compose_layers=None,
            prompt_inpaint=prompt_inpaint,
            prompt_refine=prompt_refine,
            mask_user=mask_user,
        )

    @staticmethod
    def _squeeze_to_1hw(m: torch.Tensor) -> torch.Tensor:
        return _squeeze_mask_to_1hw(m)

    @staticmethod
    def for_compose_multi(
        removal_masks: Sequence[torch.Tensor],
        compose_layers: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        prompt_inpaint: str,
        prompt_refine: str,
        mask_user: Optional[torch.Tensor] = None,
    ) -> GeometrySpec:
        """Multi-object replacement: union-inpaint then sequential paste from ``tgt``.

        Each entry in ``compose_layers`` is ``(forward_transform_3x3, paste_mask)``
        where ``paste_mask`` is ``[1, H, W]`` drawn in **unwarped target-image** pixel
        coordinates (same grid as ``tgt`` before that layer's affine). It is warped
        with ``affine_transform_mask(paste_mask, T)`` so it matches the canvas region
        where the inverse-warped target is composited. ``removal_masks`` is one mask per
        object (same length as ``compose_layers``); they are **unioned** for the first
        inpaint. ``mask_tgt`` is the union of the per-layer canvas paste masks (seam
        refinement band). The stored ``transform_matrix`` is the first layer's ``T``
        (for introspection only).
        """
        rlist = list(removal_masks)
        clist = list(compose_layers)
        if not rlist or not clist:
            raise ValueError("for_compose_multi requires non-empty removal_masks and compose_layers")
        if len(rlist) != len(clist):
            raise ValueError(
                f"for_compose_multi: len(removal_masks)={len(rlist)} must equal "
                f"len(compose_layers)={len(clist)}"
            )
        rm = [GeometrySpec._squeeze_to_1hw(m) for m in rlist]
        device, dtype = rm[0].device, rm[0].dtype
        H, W = rm[0].shape[1], rm[0].shape[2]
        union_r = rm[0].clone()
        for m in rm[1:]:
            if m.shape[1:] != (H, W):
                raise ValueError(f"All removal masks must be [1,{H},{W}], got {tuple(m.shape)}")
            union_r = torch.maximum(union_r, m.to(device=device, dtype=dtype))

        layers_norm: List[Tuple[torch.Tensor, torch.Tensor]] = []
        pastes: List[torch.Tensor] = []
        for T, pm in clist:
            if tuple(T.shape) != (3, 3):
                raise ValueError(f"Each transform must be 3x3, got {tuple(T.shape)}")
            pm = GeometrySpec._squeeze_to_1hw(pm).to(device=device, dtype=dtype)
            if pm.shape[1:] != (H, W):
                raise ValueError(f"All paste masks must be [1,{H},{W}], got {tuple(pm.shape)}")
            Tn = T.to(device=device, dtype=dtype)
            pm_canvas = affine_transform_mask(pm, Tn)
            layers_norm.append((Tn, pm_canvas))
            pastes.append(pm_canvas)

        union_p = torch.stack(pastes, dim=0).max(dim=0).values
        first_T = layers_norm[0][0]

        return GeometrySpec(
            edit_type=EditType.COMPOSE_MULTI,
            mask_src=union_r,
            mask_tgt=union_p,
            transform_matrix=first_T,
            compose_layers=layers_norm,
            prompt_inpaint=prompt_inpaint,
            prompt_refine=prompt_refine,
            mask_user=mask_user,
        )
