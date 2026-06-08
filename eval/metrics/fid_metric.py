"""Frechet Inception Distance (FID) metric."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from eval.metrics._backend import get_metric, pyiqa_available
from eval.utils import composite_for_masked_metric, to_rgb

_FID_MODEL: Any = None


class FIDAccumulator:
    """Collect reference/prediction pairs for a batch-level FID score."""

    def __init__(self) -> None:
        self._refs: list[Image.Image] = []
        self._preds: list[Image.Image] = []

    def add(
        self,
        ref: Image.Image,
        pred: Image.Image,
        mask: Image.Image | None = None,
        *,
        masked_only: bool = True,
    ) -> None:
        """Add one pair, optionally compositing outside the inpaint mask."""
        ref_rgb = to_rgb(ref)
        pred_rgb = to_rgb(pred)
        if pred_rgb.size != ref_rgb.size:
            pred_rgb = pred_rgb.resize(ref_rgb.size, Image.Resampling.LANCZOS)

        if masked_only:
            if mask is None:
                raise ValueError("mask is required when masked_only=True")
            pred_rgb = composite_for_masked_metric(ref_rgb, pred_rgb, mask)

        self._refs.append(ref_rgb)
        self._preds.append(pred_rgb)

    def __len__(self) -> int:
        return len(self._refs)

    def compute(self, device: str = "cpu") -> float:
        """Compute FID over all accumulated image pairs."""
        if len(self._refs) < 2:
            raise ValueError(
                f"FID requires at least 2 samples, got {len(self._refs)}. "
                "Use --limit 2 or higher."
            )

        if not pyiqa_available():
            raise ImportError("pyiqa is required for FID. Install eval/requirements.txt.")

        with tempfile.TemporaryDirectory(prefix="eval_fid_") as tmp:
            ref_dir = Path(tmp) / "ref"
            pred_dir = Path(tmp) / "pred"
            ref_dir.mkdir()
            pred_dir.mkdir()

            for idx, (ref_img, pred_img) in enumerate(zip(self._refs, self._preds)):
                name = f"{idx:06d}.png"
                ref_img.save(ref_dir / name)
                pred_img.save(pred_dir / name)

            metric = get_metric("fid", device=device)
            score = metric(str(pred_dir), str(ref_dir))
            if hasattr(score, "item"):
                return float(score.item())
            return float(score)


def preload(device: str = "cpu") -> None:
    """Eagerly load the FID backend."""
    if pyiqa_available():
        get_metric("fid", device=device)


def unload() -> None:
    """Release cached FID models (handled via pyiqa cache clear)."""
    global _FID_MODEL
    _FID_MODEL = None
