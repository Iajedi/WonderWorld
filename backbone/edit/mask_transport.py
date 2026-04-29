"""Optimal-transport velocity transfer for BCOT-HVE warm-start.

Builds a cost matrix between unknown and known image tokens, solves
entropic OT via log-domain Sinkhorn, and produces a barycentric
transported source velocity for the unknown region.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from ..utils.mask_ops import boundary_distance_for_known
except ImportError:
    from backbone.utils.mask_ops import boundary_distance_for_known


def build_transport_cost(
    h_unknown: torch.Tensor,
    h_known: torch.Tensor,
    pos_unknown: torch.Tensor,
    pos_known: torch.Tensor,
    mask_2d: torch.Tensor,
    lambda_pos: float = 0.1,
    lambda_bdry: float = 1.0,
    boundary_band_width: int = 2,
) -> torch.Tensor:
    """Construct the OT cost matrix ``C`` of shape ``[B, U, K]``.

    Parameters
    ----------
    h_unknown : ``[B, U, D]``
        Hidden-state embeddings for unknown tokens (from hooked layers).
    h_known : ``[B, K, D]``
        Hidden-state embeddings for known tokens.
    pos_unknown : ``[U, 2]``
        2-D (y, x) grid coordinates for unknown tokens.
    pos_known : ``[K, 2]``
        2-D (y, x) grid coordinates for known tokens.
    mask_2d : ``[H, W]``
        Binary spatial mask (1 = unknown, 0 = known).
    lambda_pos : float
        Weight for the positional distance term.
    lambda_bdry : float
        Weight for the boundary-distance penalty on donors.
    boundary_band_width : int
        Width of the known-side boundary band used for the penalty.

    Returns
    -------
    C : ``[B, U, K]``
    """
    h_u_norm = F.normalize(h_unknown.float(), dim=-1)
    h_k_norm = F.normalize(h_known.float(), dim=-1)
    C_feat = torch.cdist(h_u_norm, h_k_norm, p=2.0).pow(2)  # [B, U, K]

    C_pos = torch.cdist(
        pos_unknown.float().unsqueeze(0),
        pos_known.float().unsqueeze(0),
        p=2.0,
    ).pow(2)  # [1, U, K]

    d_bdry = boundary_distance_for_known(mask_2d, pos_known, width=boundary_band_width)
    C_bdry = d_bdry.unsqueeze(0).unsqueeze(0).expand_as(C_feat)  # [B, U, K]

    return C_feat + lambda_pos * C_pos + lambda_bdry * C_bdry


def sinkhorn_transport(
    C: torch.Tensor,
    tau: float = 0.05,
    num_iters: int = 100,
) -> torch.Tensor:
    """Solve entropic OT in log domain for numerical stability.

    Parameters
    ----------
    C : ``[B, U, K]``
        Cost matrix (non-negative).
    tau : float
        Entropy regularisation strength (lower = sharper plan).
    num_iters : int
        Number of Sinkhorn iterations.

    Returns
    -------
    pi : ``[B, U, K]``
        Row-normalised transport plan (rows sum to 1).
    """
    B, U, K = C.shape
    log_K_mat = -C / max(tau, 1e-8)  # [B, U, K]

    log_u = torch.zeros(B, U, 1, device=C.device, dtype=C.dtype)
    log_v = torch.zeros(B, 1, K, device=C.device, dtype=C.dtype)

    for _ in range(num_iters):
        log_u = -torch.logsumexp(log_K_mat + log_v, dim=-1, keepdim=True)
        log_v = -torch.logsumexp(log_K_mat + log_u, dim=-2, keepdim=True)

    log_pi = log_u + log_K_mat + log_v  # [B, U, K]
    pi = log_pi.exp()
    pi = pi / (pi.sum(dim=-1, keepdim=True) + 1e-12)
    return pi


def barycentric_velocity(
    pi: torch.Tensor,
    v_src_known: torch.Tensor,
) -> torch.Tensor:
    """Transport known-region source velocity to unknown tokens.

    Parameters
    ----------
    pi : ``[B, U, K]``
        Transport plan from :func:`sinkhorn_transport`.
    v_src_known : ``[B, K, C]``
        Source-branch velocity on known tokens.

    Returns
    -------
    v_transported : ``[B, U, C]``
    """
    return torch.bmm(pi.to(v_src_known.dtype), v_src_known)
