"""Screened-harmonic velocity extension for BCOT-HVE.

Given a transported source velocity on masked (unknown) tokens and Dirichlet
boundary conditions from neighbouring known tokens, solve a screened Poisson
system on the masked sub-grid to produce a smooth, boundary-continuous
velocity field.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    from ..utils.mask_ops import build_neighbor_pairs
except ImportError:
    from backbone.utils.mask_ops import build_neighbor_pairs


def build_masked_laplacian(
    mask_2d: torch.Tensor,
    connectivity: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the graph Laplacian restricted to masked (unknown) tokens.

    Parameters
    ----------
    mask_2d : ``[H, W]``
        Binary mask, 1 = unknown, 0 = known.
    connectivity : int
        4-neighbour or 8-neighbour grid adjacency.

    Returns
    -------
    L : ``[U, U]``
        Dense graph Laplacian on masked nodes.
    masked_flat_indices : ``[U]``
        Flat indices of unknown tokens in the full ``H*W`` grid.
    boundary_flat_indices : ``[B_n]``
        Flat indices of *known* tokens that are direct neighbours of at
        least one unknown token (Dirichlet boundary nodes).
    bdry_to_masked_adj : ``[U, B_n]``
        Adjacency matrix between unknown tokens and boundary nodes
        (1 where they are grid neighbours, 0 otherwise).
    """
    H, W = mask_2d.shape
    device = mask_2d.device

    mask_flat = mask_2d.reshape(-1) > 0.5
    masked_indices = mask_flat.nonzero(as_tuple=False).squeeze(-1)  # [U]
    U = masked_indices.numel()

    if U == 0:
        empty = torch.zeros(0, 0, device=device)
        return empty, masked_indices, torch.zeros(0, dtype=torch.long, device=device), empty

    flat_to_local = torch.full((H * W,), -1, dtype=torch.long, device=device)
    flat_to_local[masked_indices] = torch.arange(U, device=device)

    pairs = build_neighbor_pairs(H, W, connectivity).to(device)

    both_masked = mask_flat[pairs[:, 0]] & mask_flat[pairs[:, 1]]
    inner_pairs = pairs[both_masked]

    A = torch.zeros(U, U, device=device)
    if inner_pairs.numel() > 0:
        li = flat_to_local[inner_pairs[:, 0]]
        lj = flat_to_local[inner_pairs[:, 1]]
        A[li, lj] = 1.0

    D = A.sum(dim=1).diag()
    L = D - A

    one_masked = mask_flat[pairs[:, 0]] ^ mask_flat[pairs[:, 1]]
    boundary_pairs = pairs[one_masked]

    boundary_known_set: set = set()
    bdry_adj_list: list[Tuple[int, int]] = []
    for row in boundary_pairs:
        a, b = row[0].item(), row[1].item()
        if mask_flat[a] and not mask_flat[b]:
            boundary_known_set.add(b)
            bdry_adj_list.append((flat_to_local[a].item(), b))
        elif mask_flat[b] and not mask_flat[a]:
            boundary_known_set.add(a)
            bdry_adj_list.append((flat_to_local[b].item(), a))

    boundary_flat_indices = torch.tensor(sorted(boundary_known_set), dtype=torch.long, device=device)
    B_n = boundary_flat_indices.numel()

    bdry_local = torch.full((H * W,), -1, dtype=torch.long, device=device)
    if B_n > 0:
        bdry_local[boundary_flat_indices] = torch.arange(B_n, device=device)

    bdry_to_masked_adj = torch.zeros(U, B_n, device=device)
    for mi, bi_flat in bdry_adj_list:
        bj = bdry_local[bi_flat].item()
        if bj >= 0:
            bdry_to_masked_adj[mi, bj] = 1.0

    return L, masked_indices, boundary_flat_indices, bdry_to_masked_adj


def harmonic_extend(
    v_transported: torch.Tensor,
    v_tgt: torch.Tensor,
    v_src_full: torch.Tensor,
    mask_2d: torch.Tensor,
    alpha: float,
    lambda_s: float,
    token_hw: Tuple[int, int],
    connectivity: int = 4,
) -> torch.Tensor:
    """Screened-harmonic extension of the transported velocity.

    Solves per channel:

        (L_M + lambda_s I) v_tilde = lambda_s (alpha v_transport + (1-alpha) v_tgt)
                                      + B_adj @ v_bdry_src

    Boundary velocities are extracted automatically from ``v_src_full``
    using the boundary indices determined by the graph Laplacian.

    Parameters
    ----------
    v_transported : ``[B, U, C]``
        OT-transported source velocity on unknown tokens.
    v_tgt : ``[B, U, C]``
        Target-branch velocity on unknown tokens.
    v_src_full : ``[B, N, C]``
        Source-branch velocity for **all** tokens (known + unknown).
        Boundary velocities are sliced from this tensor internally.
    mask_2d : ``[H, W]``
        Binary mask (1 = unknown).
    alpha : float
        Blending weight (higher = more transported, lower = more target).
    lambda_s : float
        Screening strength (data-fidelity vs. smoothness trade-off).
    token_hw : ``(int, int)``
        Token grid height and width.
    connectivity : int
        Grid adjacency (4 or 8).

    Returns
    -------
    v_tilde : ``[B, U, C]``
        Harmonically-extended velocity for unknown tokens.
    """
    L, masked_idx, bdry_idx, B_adj = build_masked_laplacian(mask_2d, connectivity)
    U = masked_idx.numel()

    if U == 0:
        return v_transported

    B_batch, _, C = v_transported.shape
    device = v_transported.device
    dtype = v_transported.dtype

    L = L.to(device=device, dtype=torch.float32)
    B_adj = B_adj.to(device=device, dtype=torch.float32)
    A_sys = L + lambda_s * torch.eye(U, device=device, dtype=torch.float32)  # [U, U]

    data_term = lambda_s * (
        alpha * v_transported.float() + (1.0 - alpha) * v_tgt.float()
    )  # [B, U, C]

    if B_adj.numel() > 0 and bdry_idx.numel() > 0:
        v_bdry_src = v_src_full[:, bdry_idx, :].float()  # [B, B_n, C]
        bdry_term = torch.matmul(B_adj, v_bdry_src)  # [B, U, C]
    else:
        bdry_term = torch.zeros_like(data_term)

    rhs = data_term + bdry_term  # [B, U, C]

    A_batched = A_sys.unsqueeze(0).expand(B_batch, -1, -1)
    v_tilde = torch.linalg.solve(A_batched, rhs)  # [B, U, C]

    return v_tilde.to(dtype=dtype)
