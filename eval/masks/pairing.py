"""Mask pairing utilities."""

from __future__ import annotations

import numpy as np


def pair_indices(
    num_images: int,
    num_masks: int,
    *,
    mode: str = "fixed",
    seed: int = 42,
) -> list[int]:
    if num_images != num_masks:
        raise ValueError(
            f"Image count ({num_images}) must equal mask count ({num_masks}) for pairing."
        )

    if mode == "fixed":
        return list(range(num_images))

    if mode == "random":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(num_masks)
        return perm.tolist()

    raise ValueError(f"Unknown pairing mode: {mode}")
