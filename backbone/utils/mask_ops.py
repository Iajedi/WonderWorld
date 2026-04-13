"""Mask utilities for BCOT-HVE warm-start inpainting/outpainting.

Provides helpers to convert masks between pixel, latent, and token space,
extract boundary bands, build neighbour graphs, and construct 2-D coordinate
grids over the token lattice.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def infer_token_hw(token_count: int) -> Tuple[int, int]:
    """Return (H, W) for a square token grid, else (1, N)."""
    side = int(math.sqrt(token_count))
    if side * side == token_count:
        return side, side
    return 1, token_count


def build_token_grid(token_hw: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Row-major (y, x) coordinate grid of shape ``[H*W, 2]``."""
    h, w = token_hw
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([ys, xs], dim=-1).reshape(-1, 2)


def resize_mask_to_latent(
    mask: Union[np.ndarray, torch.Tensor],
    token_hw: Tuple[int, int],
) -> torch.Tensor:
    """Resize a pixel-space mask to the latent token grid.

    Accepts [H, W], [1, H, W], [H, W, 1], or [1, 1, H, W].
    Returns ``[1, 1, token_h, token_w]`` float tensor in [0, 1].
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    mask = mask.float()

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask.unsqueeze(0)
        elif mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"3-D mask must be [1,H,W] or [H,W,1], got {mask.shape}")
    elif mask.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape}")

    mask = mask.clamp(0.0, 1.0)
    th, tw = token_hw
    return F.interpolate(mask, size=(th, tw), mode="bilinear", align_corners=False)


def mask_to_token_space(
    mask: Union[np.ndarray, torch.Tensor],
    token_hw: Tuple[int, int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a pixel mask to ``[B, N, 1]`` token-space mask."""
    m = resize_mask_to_latent(mask, token_hw)  # [1,1,H,W]
    th, tw = token_hw
    m = m.reshape(1, th * tw, 1)
    if batch_size > 1:
        m = m.expand(batch_size, -1, -1)
    return m.to(device=device, dtype=dtype)


def reinject_mask_token_expanded_unknown(
    mask: Union[np.ndarray, torch.Tensor],
    token_hw: Tuple[int, int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    expand_unknown_pixels: int = 0,
) -> torch.Tensor:
    """Token mask for reinjection with an optional widened unknown (hole) region.

    When *expand_unknown_pixels* is 0, this matches :func:`mask_to_token_space`
    (bilinear resize of the continuous mask).

    When *expand_unknown_pixels* > 0, unknown is ``(mask > 0.5)`` in image
    space, dilated by that many pixels, then downsampled to the token grid with
    nearest-neighbour.  That widens the inpainted region for reinjection only,
    so tokens in the soft-mask boundary band (often dark after VAE composite)
    keep the denoised latent instead of being overwritten by inversion.
    """
    if expand_unknown_pixels <= 0:
        return mask_to_token_space(mask, token_hw, batch_size, device, dtype)

    if isinstance(mask, np.ndarray):
        m = torch.from_numpy(mask).float()
    else:
        m = mask.float().detach()

    if m.ndim == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.ndim == 3:
        if m.shape[0] == 1:
            m = m.unsqueeze(0)
        elif m.shape[-1] == 1:
            m = m.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"3-D mask must be [1,H,W] or [H,W,1], got {m.shape}")
    elif m.ndim == 4:
        if m.shape[0] != 1 or m.shape[1] != 1:
            raise ValueError(f"4-D mask must be [1,1,H,W], got {m.shape}")
    else:
        raise ValueError(f"Unsupported mask ndim {m.ndim}")

    m = m.clamp(0.0, 1.0)
    unknown = (m > 0.5).float()
    r = int(expand_unknown_pixels)
    k = 2 * r + 1
    expanded = F.max_pool2d(unknown, kernel_size=k, stride=1, padding=r)

    th, tw = token_hw
    expanded_tok = F.interpolate(expanded, size=(th, tw), mode="nearest")
    out = expanded_tok.reshape(1, th * tw, 1).clamp(0.0, 1.0)
    if batch_size > 1:
        out = out.expand(batch_size, -1, -1)
    return out.to(device=device, dtype=dtype)


def dilate_token_mask(
    mask_token: torch.Tensor,
    token_hw: Tuple[int, int],
    dilate_tokens: int,
) -> torch.Tensor:
    """Morphologically dilate the unknown (1) region on the token grid.

    ``mask_token`` is ``[1, N, 1]`` with 1 = unknown.  Each *dilate_tokens*
    step widens unknown by one token in the 4-neighbour sense via max-pool.
    """
    if dilate_tokens <= 0:
        return mask_token
    h, w = token_hw
    m_2d = mask_token[:, :, 0].reshape(1, 1, h, w).float()
    k = 2 * int(dilate_tokens) + 1
    pad = int(dilate_tokens)
    m_2d = F.max_pool2d(m_2d, kernel_size=k, stride=1, padding=pad)
    return m_2d.reshape(1, h * w, 1).to(mask_token.dtype)


def mask_2d_from_token_mask(
    mask_token: torch.Tensor,
    token_hw: Tuple[int, int],
) -> torch.Tensor:
    """Reshape a flat ``[…, N, 1]`` or ``[N]`` token mask to ``[H, W]``."""
    m = mask_token.detach().float()
    if m.ndim >= 3:
        m = m[0, :, 0]
    elif m.ndim == 2:
        m = m[:, 0]
    return m.reshape(token_hw)


def extract_boundary_band(mask_2d: torch.Tensor, width: int = 2) -> torch.Tensor:
    """Return a binary mask of *known* tokens within ``width`` of the mask edge.

    ``mask_2d`` is ``[H, W]`` with 1 = unknown, 0 = known.
    Output is ``[H, W]`` with 1 = known-boundary, 0 = elsewhere.
    """
    m = mask_2d.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    kernel_size = 2 * width + 1
    dilated = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=width)
    dilated = dilated.squeeze(0).squeeze(0)
    boundary = (dilated > 0.5) & (mask_2d < 0.5)
    return boundary.float()


def get_boundary_token_indices(mask_2d: torch.Tensor, width: int = 2) -> torch.Tensor:
    """Flat indices of known tokens in the boundary band."""
    band = extract_boundary_band(mask_2d, width)
    return band.reshape(-1).nonzero(as_tuple=False).squeeze(-1)


def boundary_distance_for_known(
    mask_2d: torch.Tensor,
    known_pos: torch.Tensor,
    width: int = 2,
) -> torch.Tensor:
    """For each known token, compute its distance to the nearest mask boundary.

    Returns ``[K]`` tensor (0 for tokens on the boundary, larger farther away).
    Used inside the OT cost matrix to penalise donors far from the edge.
    """
    band = extract_boundary_band(mask_2d, width)
    bdry_idx = band.reshape(-1).nonzero(as_tuple=False).squeeze(-1)

    if bdry_idx.numel() == 0:
        return torch.zeros(known_pos.shape[0], device=known_pos.device)

    h, w = mask_2d.shape
    ys = bdry_idx // w
    xs = bdry_idx % w
    bdry_pos = torch.stack([ys.float(), xs.float()], dim=-1).to(known_pos.device)

    dists = torch.cdist(known_pos.float(), bdry_pos, p=2.0)  # [K, B_band]
    return dists.min(dim=-1).values


def build_neighbor_pairs(H: int, W: int, connectivity: int = 4) -> torch.Tensor:
    """Return ``[E, 2]`` int tensor of (flat_i, flat_j) neighbor pairs.

    ``connectivity`` is 4 (rook) or 8 (king).
    """
    pairs = []
    for y in range(H):
        for x in range(W):
            idx = y * W + x
            if connectivity >= 4:
                if x + 1 < W:
                    pairs.append((idx, idx + 1))
                if y + 1 < H:
                    pairs.append((idx, idx + W))
            if connectivity >= 8:
                if x + 1 < W and y + 1 < H:
                    pairs.append((idx, idx + W + 1))
                if x - 1 >= 0 and y + 1 < H:
                    pairs.append((idx, idx + W - 1))
    if not pairs:
        return torch.zeros(0, 2, dtype=torch.long)
    t = torch.tensor(pairs, dtype=torch.long)
    sym = torch.cat([t, t.flip(1)], dim=0)
    return sym
