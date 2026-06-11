"""Poisson interpolation (a form of harmonic extension), used for BCDM warm start.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    from ..utils.mask_ops import build_neighbor_pairs
except ImportError:
    from backbone.utils.mask_ops import build_neighbor_pairs


# Helper function to build the graph Laplacian restricted to masked (unknown) tokens. Writing assisted by Cursor Claude Opus 4.6
def build_masked_laplacian(
    mask_2d: torch.Tensor,
    connectivity: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Get mask dimensions and device
    H, W = mask_2d.shape
    device = mask_2d.device

    # Flatten mask and get masked indices
    mask_flat = mask_2d.reshape(-1) > 0.5
    masked_indices = mask_flat.nonzero(as_tuple=False).squeeze(-1)  # [U]
    U = masked_indices.numel()

    if U == 0:
        empty = torch.zeros(0, 0, device=device)
        return empty, masked_indices, torch.zeros(0, dtype=torch.long, device=device), empty

    flat_to_local = torch.full((H * W,), -1, dtype=torch.long, device=device)
    flat_to_local[masked_indices] = torch.arange(U, device=device)

    # Build neighbor pairs (8-neighbour grid adjacency)
    pairs = build_neighbor_pairs(H, W, connectivity).to(device)

    both_masked = mask_flat[pairs[:, 0]] & mask_flat[pairs[:, 1]]
    inner_pairs = pairs[both_masked]

    A = torch.zeros(U, U, device=device)
    # Build adjacency matrix for inner pairs
    if inner_pairs.numel() > 0:
        # Get local indices of inner pairs (inner pairs, set to 1.0)
        li = flat_to_local[inner_pairs[:, 0]]
        lj = flat_to_local[inner_pairs[:, 1]]
        A[li, lj] = 1.0

    # Build diagonal degree matrix
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


# Core poisson interpolation (harmonic extension) function.
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
    # Build masked Laplacian
    # We 
    L, masked_idx, bdry_idx, B_adj = build_masked_laplacian(mask_2d, connectivity)
    U = masked_idx.numel()

    if U == 0:
        return v_transported

    B_batch, _, C = v_transported.shape
    device = v_transported.device
    dtype = v_transported.dtype

    L = L.to(device=device, dtype=torch.float32)
    B_adj = B_adj.to(device=device, dtype=torch.float32)
    
    # Build coefficient matrix (L + lambda_s * I)
    A_sys = L + lambda_s * torch.eye(U, device=device, dtype=torch.float32)  # [U, U]

    # Build lambda_s * u_t (rhs of equation) (transported + target velocity)
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

    # Solve for harmonised unknown region velocity
    v_tilde = torch.linalg.solve(A_batched, rhs)  # [B, U, C]

    return v_tilde.to(dtype=dtype)
