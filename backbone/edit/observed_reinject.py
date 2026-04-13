"""Observed-region token reinjection from the inversion trajectory.

At every post-warm-start denoising step, the observed (known) region of
the current latent is hard-replaced with the corresponding tokens from
the stored inversion trajectory.  This prevents colorimetric drift in
the observed region while allowing the unknown region to evolve freely.
"""

from __future__ import annotations

import torch


def reinject_observed_tokens(
    z_cur: torch.Tensor,
    z_inv: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Replace observed-region tokens with inversion-trajectory tokens.

    Parameters
    ----------
    z_cur : ``[B, N, C]``
        Current latent in packed token format.
    z_inv : ``[B, N, C]``
        Inversion-trajectory latent for the **same timestep**, packed.
    mask : ``[1, N, 1]`` or broadcastable
        Token mask where 1 = unknown, 0 = observed/known.
        No feathering; used as-is.

    Returns
    -------
    z_out : ``[B, N, C]``
        Latent with observed region from ``z_inv``, unknown from ``z_cur``.
    """
    return mask * z_cur + (1.0 - mask) * z_inv
