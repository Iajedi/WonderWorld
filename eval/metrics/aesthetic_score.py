"""CLIP Aesthetic Score (CAS) using LAION improved aesthetic predictor."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from eval.metrics.clip_consistency import encode_image

_AESTHETIC_MLP: nn.Module | None = None
_AESTHETIC_DEVICE: str | None = None

AESTHETIC_REPO = "camenduru/improved-aesthetic-predictor"
AESTHETIC_WEIGHTS = "sac+logos+ava1-l14-linearMSE.pth"


class _AestheticMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_aesthetic_mlp(device: str) -> nn.Module:
    """Download and load the LAION aesthetic MLP head."""
    global _AESTHETIC_MLP, _AESTHETIC_DEVICE
    if _AESTHETIC_MLP is None or _AESTHETIC_DEVICE != device:
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(
            repo_id=AESTHETIC_REPO,
            filename=AESTHETIC_WEIGHTS,
        )
        mlp = _AestheticMLP()
        state_dict = torch.load(weights_path, map_location="cpu")
        mlp.load_state_dict(state_dict)
        _AESTHETIC_MLP = mlp.to(device).eval()
        _AESTHETIC_DEVICE = device
    return _AESTHETIC_MLP


def compute_aesthetic_scores(
    views: list[Image.Image],
    clip_model: Any,
    clip_preprocess: Any,
    aesthetic_mlp: nn.Module,
    device: str,
) -> dict:
    """
    Encode each view with CLIP ViT-L/14, L2-normalise, pass through aesthetic MLP.

    Returns:
        per_view_cas: list of float
        mean_cas: float
    """
    per_view_cas: list[float] = []
    with torch.no_grad():
        for view in views:
            emb = encode_image(clip_model, clip_preprocess, view, device)
            score = aesthetic_mlp(emb.unsqueeze(0)).squeeze()
            per_view_cas.append(float(score.item()))

    mean_cas = float(sum(per_view_cas) / len(per_view_cas)) if per_view_cas else float("nan")
    return {"per_view_cas": per_view_cas, "mean_cas": mean_cas}


def preload(device: str = "cpu") -> nn.Module:
    """Eagerly load the aesthetic MLP."""
    return load_aesthetic_mlp(device)


def unload() -> None:
    """Release cached aesthetic MLP."""
    global _AESTHETIC_MLP, _AESTHETIC_DEVICE
    _AESTHETIC_MLP = None
    _AESTHETIC_DEVICE = None
