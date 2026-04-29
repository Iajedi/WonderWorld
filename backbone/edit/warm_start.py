"""K-step warm-start loop with transformer hook management for BCOT-HVE.

Registers forward hooks on selected DiT blocks to capture intermediate
hidden states, runs K denoising steps with OT-based velocity transport
and harmonic extension, then returns the warmed latent for the standard
UniEdit-Flow phase.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from .mask_transport import (
        barycentric_velocity,
        build_transport_cost,
        sinkhorn_transport,
    )
    from .harmonic_extension import harmonic_extend
    from .baselines import kv_replace_warm_step, masked_attention_warm_step
    from ..utils.mask_ops import (
        build_token_grid,
        mask_2d_from_token_mask,
    )
except ImportError:
    from backbone.edit.mask_transport import (
        barycentric_velocity,
        build_transport_cost,
        sinkhorn_transport,
    )
    from backbone.edit.harmonic_extension import harmonic_extend
    from backbone.edit.baselines import kv_replace_warm_step, masked_attention_warm_step
    from backbone.utils.mask_ops import (
        build_token_grid,
        mask_2d_from_token_mask,
    )


class WarmStartHookManager:
    """Register / deregister forward hooks on Flux2 transformer blocks.

    Captured hidden states are stored per ``(block_type, index)`` key and
    can be queried after each transformer forward pass.

    Parameters
    ----------
    transformer : ``Flux2Transformer2DModel``
        The transformer sub-module of the Flux pipeline.
    warm_layers : list of (str, int)
        Each entry is ``("double", i)`` or ``("single", j)`` identifying a
        block to hook.
    num_txt_tokens : int
        Length of the text-token prefix in the single-stream blocks
        (needed to strip text tokens from the captured output).
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        warm_layers: List[Tuple[str, int]],
        num_txt_tokens: int,
    ):
        self.transformer = transformer
        self.warm_layers = warm_layers
        self.num_txt_tokens = num_txt_tokens
        self._handles: list = []
        self._captured: Dict[Tuple[str, int], torch.Tensor] = {}

    def register(self) -> None:
        for block_type, idx in self.warm_layers:
            if block_type == "double":
                block = self.transformer.transformer_blocks[idx]
            elif block_type == "single":
                block = self.transformer.single_transformer_blocks[idx]
            else:
                raise ValueError(f"Unknown block type {block_type!r}")

            key = (block_type, idx)
            handle = block.register_forward_hook(self._make_hook(key, block_type))
            self._handles.append(handle)

    def _make_hook(self, key: Tuple[str, int], block_type: str):
        def hook_fn(module, inp, output):
            if block_type == "double":
                # output = (encoder_hidden_states, hidden_states)
                img_hidden = output[1].detach()
            else:
                # output = combined [B, N_txt + N_img, D]
                if isinstance(output, tuple):
                    combined = output[0].detach()
                else:
                    combined = output.detach()
                img_hidden = combined[:, self.num_txt_tokens:, :]
            self._captured[key] = img_hidden
        return hook_fn

    def get_image_hidden_states(self, block_type: str, idx: int) -> torch.Tensor:
        """Return ``[B, N_img, D]`` image-only hidden states for a block."""
        return self._captured[(block_type, idx)]

    def get_averaged_hidden_states(self) -> torch.Tensor:
        """Average captured image hidden states across all hooked layers."""
        tensors = list(self._captured.values())
        if len(tensors) == 1:
            return tensors[0]
        return torch.stack(tensors, dim=0).mean(dim=0)

    def clear(self) -> None:
        self._captured.clear()

    def deregister(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._captured.clear()


def _alpha_schedule_linear(step: int, K: int, alpha_start: float, alpha_end: float) -> float:
    if K <= 1:
        return alpha_start
    return alpha_start + (alpha_end - alpha_start) * step / (K - 1)


@torch.no_grad()
def warm_start_loop(
    pipe: Any,
    z_t: torch.Tensor,
    mask_token: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    sigmas_warm: torch.Tensor,
    warm_layers: List[Tuple[str, int]],
    config: Dict[str, Any],
    warm_method: str = "ot_harmonic",
    debug_dir: Optional[str] = None,
) -> torch.Tensor:
    """Run K warm-start denoising steps on the unknown region.

    Parameters
    ----------
    pipe : Flux2KleinPipeline (or FluxPipeline)
        The underlying diffusers pipeline (``pipe.pipe`` from the controller).
    z_t : ``[2, N, C]``
        Initial latent (batch-doubled: source row 0, target row 1).
    mask_token : ``[1, N, 1]``
        Token-space mask, 1 = unknown, 0 = known.
    prompt_embeds : ``[2, S, D]``
        Text embeddings for source and target prompts.
    text_ids : ``[2, S, 4]`` or ``[S, 4]``
        Positional IDs for text tokens.
    latent_image_ids : ``[2, N, 4]`` or ``[N, 4]``
        Positional IDs for image tokens.
    sigmas_warm : ``[K+1]``
        Sigma values for the K warm-start intervals.
    warm_layers : list of ``("double"|"single", int)``
        Transformer blocks to hook.
    config : dict
        Hyperparameters (see ``configs/*.yaml``).
    warm_method : str
        One of ``none``, ``kv_replace``, ``masked_attention``,
        ``ot_only``, ``ot_harmonic``.
    debug_dir : str or None
        If set, save per-step debug outputs here.

    Returns
    -------
    z_t : ``[2, N, C]``
        Warmed latent ready for the standard UniEdit-Flow phase.
    """
    K = len(sigmas_warm) - 1
    if K <= 0:
        return z_t

    device = z_t.device
    dtype = z_t.dtype
    N = z_t.shape[1]

    try:
        from ..utils.mask_ops import infer_token_hw
    except ImportError:
        from backbone.utils.mask_ops import infer_token_hw
    token_hw = config.get("token_hw") or infer_token_hw(N)
    H, W = token_hw

    mask_2d = mask_2d_from_token_mask(mask_token, token_hw).to(device)
    mask_flat = mask_token[0, :, 0] > 0.5  # [N]
    known_flat = ~mask_flat

    coords = build_token_grid(token_hw, device=device, dtype=torch.float32)

    unk_idx = mask_flat.nonzero(as_tuple=False).squeeze(-1)
    kn_idx = known_flat.nonzero(as_tuple=False).squeeze(-1)
    pos_unk = coords[unk_idx]
    pos_kn = coords[kn_idx]

    num_txt = prompt_embeds.shape[1]
    hook_mgr: Optional[WarmStartHookManager] = None
    if warm_method in ("ot_only", "ot_harmonic"):
        hook_mgr = WarmStartHookManager(pipe.transformer, warm_layers, num_txt)
        hook_mgr.register()

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    m = mask_token.to(device=device, dtype=dtype)  # [1, N, 1]

    try:
        for k in range(K):
            sigma = sigmas_warm[k]
            sigma_next = sigmas_warm[k + 1]
            dt = sigma_next - sigma  # negative (descending)

            timestep = (sigma * 1000.0).expand(z_t.shape[0]).to(dtype)

            if hook_mgr is not None:
                hook_mgr.clear()

            noise_pred = pipe.transformer(
                hidden_states=z_t.to(pipe.transformer.dtype),
                timestep=timestep / 1000.0,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            noise_pred = noise_pred[:, :N, :]

            v_src, v_trg = noise_pred.chunk(2, dim=0)  # each [1, N, C]

            alpha_k = _alpha_schedule_linear(
                k, K,
                config.get("alpha_start", 0.9),
                config.get("alpha_end", 0.3),
            )

            if warm_method == "none":
                v_unknown = v_trg[:, unk_idx, :]

            elif warm_method == "kv_replace":
                v_warm_full = kv_replace_warm_step(v_src, v_trg, m, token_hw)
                v_unknown = v_warm_full[:, unk_idx, :]

            elif warm_method == "masked_attention":
                v_warm_full = masked_attention_warm_step(v_src, v_trg, m, token_hw)
                v_unknown = v_warm_full[:, unk_idx, :]

            elif warm_method in ("ot_only", "ot_harmonic"):
                h_avg = hook_mgr.get_averaged_hidden_states()  # [2, N, D]
                h_src_all, h_tgt_all = h_avg.chunk(2, dim=0)  # each [1, N, D]

                h_known = h_src_all[:, kn_idx, :]   # [1, K_n, D]
                h_unknown = h_tgt_all[:, unk_idx, :]  # [1, U, D]

                C = build_transport_cost(
                    h_unknown=h_unknown,
                    h_known=h_known,
                    pos_unknown=pos_unk,
                    pos_known=pos_kn,
                    mask_2d=mask_2d,
                    lambda_pos=config.get("lambda_pos", 0.1),
                    lambda_bdry=config.get("lambda_bdry", 1.0),
                    boundary_band_width=config.get("boundary_band_width", 2),
                )

                pi = sinkhorn_transport(
                    C,
                    tau=config.get("tau", 0.05),
                    num_iters=config.get("sinkhorn_iters", 100),
                )

                v_src_known = v_src[:, kn_idx, :]  # [1, K_n, C]
                v_transported = barycentric_velocity(pi, v_src_known)  # [1, U, C]

                if warm_method == "ot_only":
                    v_tgt_unk = v_trg[:, unk_idx, :]
                    v_unknown = alpha_k * v_transported + (1.0 - alpha_k) * v_tgt_unk
                else:
                    v_tgt_unk = v_trg[:, unk_idx, :]

                    v_unknown = harmonic_extend(
                        v_transported=v_transported,
                        v_tgt=v_tgt_unk,
                        v_src_full=v_src,
                        mask_2d=mask_2d,
                        alpha=alpha_k,
                        lambda_s=config.get("lambda_s", 0.5),
                        token_hw=token_hw,
                        connectivity=config.get("connectivity", 4),
                    )

                if debug_dir is not None:
                    _save_warm_debug(debug_dir, k, pi, mask_2d, v_unknown, token_hw)
            else:
                raise ValueError(f"Unknown warm_method {warm_method!r}")

            v_full = torch.zeros_like(v_src)  # [1, N, C]
            v_full[:, unk_idx, :] = v_unknown.to(dtype)
            v_full_batch = v_full.expand(2, -1, -1)  # both branches get same update

            z_t = z_t.float()
            z_t = (1.0 - m) * z_t + m * (z_t + dt * v_full_batch.float())
            z_t = z_t.to(dtype)

    finally:
        if hook_mgr is not None:
            hook_mgr.deregister()

    return z_t


def _save_warm_debug(
    debug_dir: str,
    step: int,
    pi: Optional[torch.Tensor],
    mask_2d: torch.Tensor,
    v_unknown: torch.Tensor,
    token_hw: Tuple[int, int],
) -> None:
    """Save lightweight per-step debug tensors."""
    prefix = os.path.join(debug_dir, f"warm_step_{step:03d}")
    if pi is not None:
        torch.save(pi.cpu(), f"{prefix}_transport_plan.pt")
    torch.save(mask_2d.cpu(), f"{prefix}_mask_2d.pt")
    v_norm = v_unknown.float().norm(dim=-1).cpu()
    torch.save(v_norm, f"{prefix}_v_unknown_norm.pt")
