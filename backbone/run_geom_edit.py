#!/usr/bin/env python
"""Run a single EditPipeline (geometry) example end-to-end.

Usage
-----
    cd backbone
    python run_geom_edit.py [--config configs/geom_edit_pipeline.yaml] [--outdir outputs/geom_edit]
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

from backbone.edit.geom_controller import EditPipeline
from geometry.spec import EditType, GeometrySpec


def _paste_mask_centroid(paste_m: torch.Tensor) -> tuple[float, float]:
    """Mean ``(cx, cy)`` of foreground (``> 0.5``) in ``paste_m`` ``[1, H, W]``; image centre if empty."""
    m = paste_m.detach()
    while m.ndim > 3:
        m = m.squeeze(0)
    if m.ndim == 3:
        plane = m[0]
    else:
        plane = m
    H, W = plane.shape
    fg = plane > 0.5
    ys, xs = torch.where(fg)
    if ys.numel() == 0:
        return (W - 1) * 0.5, (H - 1) * 0.5
    return float(xs.float().mean()), float(ys.float().mean())


def _affine_from_compose_entry(entry: Dict[str, Any], paste_m: torch.Tensor) -> torch.Tensor:
    """3×3 forward map: optional ``matrix``, else ``scale`` + ``dx`` / ``dy``.

    Without ``matrix``: uniform ``scale`` (default ``1``) about the **paste_mask**
    foreground centroid, then translation by ``dx`` / ``dy`` (defaults ``0``), same
    convention as :meth:`geometry.spec.GeometrySpec.for_resize` combined with move.
    """
    if "matrix" in entry and entry["matrix"] is not None:
        m = entry["matrix"]
        if not isinstance(m, Sequence) or len(m) != 3:
            raise ValueError("'matrix' must be a 3×3 nested sequence")
        t = torch.tensor(m, dtype=torch.float32)
        if t.shape != (3, 3):
            raise ValueError("'matrix' must be 3×3")
        return t

    s = float(entry["scale"]) if "scale" in entry else 1.0
    if s <= 0.0:
        raise ValueError(f"compose scale must be positive, got {s}")
    dx = float(entry["dx"]) if "dx" in entry else 0.0
    dy = float(entry["dy"]) if "dy" in entry else 0.0

    cx, cy = _paste_mask_centroid(paste_m)
    tx = cx * (1.0 - s) + dx
    ty = cy * (1.0 - s) + dy
    return torch.tensor([[s, 0.0, tx], [0.0, s, ty], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _load_mask_tensor(path: str, size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    """``[1, H, W]`` float in ``[0, 1]`` (1 = unknown / inpaint region)."""
    h, w = size
    img = Image.open(path).convert("L").resize((w, h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _geometry_spec_from_yaml(geom: Dict[str, Any]) -> GeometrySpec:
    edit_type = EditType(str(geom["edit_type"]).lower())
    pi = str(geom["prompt_inpaint"])
    pr = geom.get("prompt_refine")
    if pr is None or (isinstance(pr, str) and not pr.strip()):
        pr_str: str | None = None
    else:
        pr_str = str(pr).strip()

    if edit_type == EditType.COMPOSE_MULTI:
        rpaths = geom.get("removal_masks")
        citems = geom.get("compose")
        if not isinstance(rpaths, list) or not rpaths:
            raise ValueError("compose_multi requires geometry.removal_masks: non-empty list of paths")
        if not isinstance(citems, list) or not citems:
            raise ValueError("compose_multi requires geometry.compose: non-empty list of layer dicts")
        if len(rpaths) != len(citems):
            raise ValueError(
                f"compose_multi: len(removal_masks)={len(rpaths)} must equal len(compose)={len(citems)}"
            )
        removal_masks: List[torch.Tensor] = [_load_mask_tensor(str(p)) for p in rpaths]
        compose_layers: List[tuple[torch.Tensor, torch.Tensor]] = []
        for entry in citems:
            if not isinstance(entry, dict):
                raise ValueError("each geometry.compose item must be a mapping")
            pm_path = entry.get("paste_mask")
            if not pm_path:
                raise ValueError("each compose layer needs 'paste_mask' (path)")
            paste_m = _load_mask_tensor(str(pm_path))
            T = _affine_from_compose_entry(entry, paste_m)
            compose_layers.append((T, paste_m))
        return GeometrySpec.for_compose_multi(
            removal_masks,
            compose_layers,
            pi,
            pr_str or pi,
            mask_user=None,
        )

    mask_src = _load_mask_tensor(geom["mask_src"])

    if edit_type == EditType.MOVE:
        spec = GeometrySpec.for_move(
            mask_src,
            float(geom.get("dx", 0.0)),
            float(geom.get("dy", 0.0)),
            pi,
            pr_str or pi,
            mask_user=None,
        )
    elif edit_type == EditType.COPY:
        spec = GeometrySpec.for_copy(
            mask_src,
            float(geom.get("dx", 0.0)),
            float(geom.get("dy", 0.0)),
            pi,
            pr_str or pi,
            mask_user=None,
        )
    elif edit_type == EditType.RESIZE:
        spec = GeometrySpec.for_resize(
            mask_src,
            float(geom.get("scale", 1.0)),
            pi,
            pr_str or pi,
            mask_user=None,
        )
    else:
        raise ValueError(f"Unknown edit_type: {edit_type}")

    return spec


def _split_pipeline_config(raw: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    geom = raw.get("geometry")
    if not isinstance(geom, dict):
        raise ValueError("config must include a 'geometry' mapping")
    paths = raw.get("inputs")
    if not isinstance(paths, dict):
        raise ValueError("config must include an 'inputs' mapping (src, optional tgt)")
    pipe_cfg = {k: v for k, v in raw.items() if k not in ("geometry", "inputs")}
    return paths, geom, pipe_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="EditPipeline single geometry run")
    parser.add_argument("--config", type=str, default="configs/geom_edit_pipeline.yaml")
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Override inputs.src from the config",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        default=None,
        help="Override inputs.tgt from the config",
    )
    parser.add_argument("--outdir", type=str, default="outputs/geom_edit")
    parser.add_argument("--offload", action="store_true", help="Enable CPU offload")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    paths, geom, pipe_cfg = _split_pipeline_config(raw)
    src_path = args.src or paths.get("src")
    tgt_path = args.tgt or paths.get("tgt")
    if not src_path:
        raise ValueError("inputs.src (or --src) is required")
    if not tgt_path:
        tgt_path = src_path

    print(f"[run_geom_edit] Config: {args.config}")
    print(f"[run_geom_edit] edit_type={geom.get('edit_type')}, warm_method={pipe_cfg.get('warm_method')}")

    spec = _geometry_spec_from_yaml(geom)
    src_image = Image.open(src_path).convert("RGB")
    tgt_image = Image.open(tgt_path).convert("RGB")

    os.makedirs(args.outdir, exist_ok=True)
    pipe = EditPipeline(offload=args.offload)
    result = pipe.run(
        src_image=src_image,
        tgt_image=tgt_image,
        spec=spec,
        config=pipe_cfg,
        output_dir=args.outdir,
    )

    m = spec.mask_src.detach().float().cpu().numpy()
    while m.ndim > 2:
        m = m.squeeze(0)
    mask_np = (m > 0.5).astype(np.float32).reshape(1, 1, 512, 512)
    metrics = pipe.compute_metrics(src_image, result, mask_np)

    print(f"[run_geom_edit] Done. Output in {args.outdir}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
