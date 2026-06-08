"""GeometrySpec builder for GeoBench FLUX evaluation (socket-style, masks as-is)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

try:
    from backbone.geometry.spec import GeometrySpec
except ImportError:
    from eval.data.geobench_geometry import GeometrySpec  # type: ignore[assignment]

from eval.utils import to_grayscale


def _mask_to_tensor(mask: Image.Image, size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    w, h = size
    gray = to_grayscale(mask).resize((w, h), Image.Resampling.NEAREST)
    arr = np.asarray(gray, dtype=np.uint8)
    binary = (arr >= 128).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)


def build_geobench_geometry_spec(
    ori_mask: Image.Image,
    tgt_mask: Image.Image,
    prompt: str,
    *,
    size: int = 512,
    removes_source_region: bool = True,
) -> GeometrySpec:
    """Build a compose_multi spec using provided masks without edit_param warps.

    Mirrors :func:`backbone.edit.socket_edit.build_socket_geometry_spec` with
    identity transform — masks are fed as-is after resize to ``size``.
    """
    source_mask_tensor = _mask_to_tensor(ori_mask, (size, size))
    target_mask_tensor = _mask_to_tensor(tgt_mask, (size, size))
    identity = torch.eye(3, dtype=torch.float32)

    if removes_source_region:
        removal_mask = torch.maximum(source_mask_tensor, target_mask_tensor)
    else:
        removal_mask = target_mask_tensor

    return GeometrySpec.for_compose_multi(
        removal_masks=[removal_mask],
        compose_layers=[(identity, target_mask_tensor)],
        prompt_inpaint=prompt,
        prompt_refine=prompt,
        mask_user=None,
    )
