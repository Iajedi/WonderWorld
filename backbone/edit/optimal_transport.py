"""Optimal-transport velocity transfer for BCDM warm-start."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from utils.mask_ops import boundary_distance_for_known
except ImportError:
    from backbone.utils.mask_ops import boundary_distance_for_known

# Construct the cost matrix
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
    # Hidden state for unknown and known tokens
    h_u_norm = F.normalize(h_unknown.float(), dim=-1)
    h_k_norm = F.normalize(h_known.float(), dim=-1)
    C_feat = torch.cdist(h_u_norm, h_k_norm, p=2.0).pow(2)  # [B, U, K]

    # Distance calculation between unknown and known tokens
    C_pos = torch.cdist(
        pos_unknown.float().unsqueeze(0),
        pos_known.float().unsqueeze(0),
        p=2.0,
    ).pow(2)  # [1, U, K]

    # Boundary distance calculation
    d_bdry = boundary_distance_for_known(mask_2d, pos_known, width=boundary_band_width)
    C_bdry = d_bdry.unsqueeze(0).unsqueeze(0).expand_as(C_feat)  # [B, U, K]

    # Cost matrix
    return C_feat + lambda_pos * C_pos + lambda_bdry * C_bdry


# Carries out Sinkhorn transport algorithm
# Adapted from original paper DOI: 10.1214/aoms/1177703591
def sinkhorn_transport(
    C: torch.Tensor,
    tau: float = 0.05,
    num_iters: int = 100,
) -> torch.Tensor:
    B, U, K = C.shape
    log_K_mat = -C / max(tau, 1e-8)  # [B, U, K]

    log_u = torch.zeros(B, U, 1, device=C.device, dtype=C.dtype)
    log_v = torch.zeros(B, 1, K, device=C.device, dtype=C.dtype)

    # The iteration steps of Sinkhorn transport
    for _ in range(num_iters):
        log_u = -torch.logsumexp(log_K_mat + log_v, dim=-1, keepdim=True)
        log_v = -torch.logsumexp(log_K_mat + log_u, dim=-2, keepdim=True)

    log_pi = log_u + log_K_mat + log_v  # [B, U, K]
    pi = log_pi.exp()
    pi = pi / (pi.sum(dim=-1, keepdim=True) + 1e-12)
    return pi
