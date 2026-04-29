"""Baseline warm-start methods for ablation comparison.

These are simpler alternatives to the full BCOT-HVE pipeline and are
selected via the ``warm_method`` config flag.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

try:
    from ..utils.mask_ops import build_token_grid
except ImportError:
    from backbone.utils.mask_ops import build_token_grid


def kv_replace_warm_step(
    v_src: torch.Tensor,
    v_trg: torch.Tensor,
    mask_token: torch.Tensor,
    token_hw: Tuple[int, int],
    k_neighbours: int = 4,
) -> torch.Tensor:
    """Replace unknown-token velocity with the mean of nearest known-token source velocity.

    This is the simplest spatial-interpolation baseline (no OT, no harmonic).

    Parameters
    ----------
    v_src : ``[B, N, C]``
        Source-branch velocity for all tokens.
    v_trg : ``[B, N, C]``
        Target-branch velocity for all tokens (unused except as fallback).
    mask_token : ``[B, N, 1]``
        Token mask, 1 = unknown, 0 = known.
    token_hw : ``(int, int)``
        Token grid spatial dimensions.
    k_neighbours : int
        Number of nearest known tokens to average.

    Returns
    -------
    v_warm : ``[B, N, C]``
        Velocity field with unknown tokens replaced.
    """
    B, N, C = v_src.shape
    device = v_src.device
    dtype = v_src.dtype

    mask_bool = mask_token[..., 0] > 0.5  # [B, N]
    known_bool = ~mask_bool

    coords = build_token_grid(token_hw, device=device, dtype=torch.float32)  # [N, 2]
    v_out = v_src.clone()

    for b in range(B):
        unk_idx = mask_bool[b].nonzero(as_tuple=False).squeeze(-1)
        kn_idx = known_bool[b].nonzero(as_tuple=False).squeeze(-1)
        if unk_idx.numel() == 0 or kn_idx.numel() == 0:
            continue

        d2 = torch.cdist(coords[unk_idx], coords[kn_idx], p=2.0)
        k_eff = min(k_neighbours, kn_idx.numel())
        _, topk_idx = d2.topk(k_eff, dim=-1, largest=False)

        gathered = v_src[b, kn_idx][topk_idx]  # [U, k, C]
        v_out[b, unk_idx] = gathered.mean(dim=1).to(dtype)

    return v_out


def masked_attention_warm_step(
    v_src: torch.Tensor,
    v_trg: torch.Tensor,
    mask_token: torch.Tensor,
    token_hw: Tuple[int, int],
    tau_space: float = 4.0,
    w_feat: float = 1.0,
    w_space: float = 1.0,
) -> torch.Tensor:
    """Re-weight unknown velocity using soft attention to known tokens only.

    Queries are ``v_trg`` at unknown positions; keys / values are ``v_src``
    at known positions.  Attention logits combine cosine feature similarity
    and spatial proximity.

    Parameters
    ----------
    v_src, v_trg : ``[B, N, C]``
    mask_token : ``[B, N, 1]``
    token_hw : ``(int, int)``
    tau_space : float
        Spatial temperature (higher = wider spatial reach).
    w_feat, w_space : float
        Weights for feature-similarity and spatial terms.

    Returns
    -------
    v_warm : ``[B, N, C]``
    """
    B, N, C = v_src.shape
    device = v_src.device
    dtype = v_src.dtype

    mask_bool = mask_token[..., 0] > 0.5
    known_bool = ~mask_bool

    coords = build_token_grid(token_hw, device=device, dtype=torch.float32)
    v_out = v_src.clone()

    for b in range(B):
        unk_idx = mask_bool[b].nonzero(as_tuple=False).squeeze(-1)
        kn_idx = known_bool[b].nonzero(as_tuple=False).squeeze(-1)
        if unk_idx.numel() == 0 or kn_idx.numel() == 0:
            continue

        q = F.normalize(v_trg[b, unk_idx].float(), dim=-1)
        k = F.normalize(v_src[b, kn_idx].float(), dim=-1)

        feat_sim = q @ k.T  # [U, K]

        dist2 = torch.cdist(coords[unk_idx], coords[kn_idx], p=2.0).pow(2)
        spatial = -dist2 / max(tau_space, 1e-6)

        logits = w_feat * feat_sim + w_space * spatial
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn)

        v_out[b, unk_idx] = (attn @ v_src[b, kn_idx].float()).to(dtype)

    return v_out
