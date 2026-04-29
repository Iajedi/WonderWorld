"""Reusable editing-session primitives.

Extracts Phase A (inversion + packing), Phase C (UniEdit-Flow editing),
and Phase D (decode) from BCOTHVEPipeline so they can be composed by
both the original BCOT-HVE controller and the unified geometric pipeline.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .observed_reinject import reinject_observed_tokens
    from ..utils.mask_ops import mask_to_token_space
except ImportError:
    from backbone.edit.observed_reinject import reinject_observed_tokens
    from backbone.utils.mask_ops import mask_to_token_space

device = "cuda"
dtype = torch.bfloat16


def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def dilate_token_mask(
    mask_token: torch.Tensor,
    token_hw: Tuple[int, int],
    dilate_pixels: int,
) -> torch.Tensor:
    """Dilate the unknown region of a ``[1, N, 1]`` token mask."""
    if dilate_pixels <= 0:
        return mask_token
    H, W = token_hw
    m_2d = mask_token[:, :, 0].reshape(1, 1, H, W).float()
    k = 2 * dilate_pixels + 1
    m_2d = F.max_pool2d(m_2d, kernel_size=k, stride=1, padding=dilate_pixels)
    return m_2d.reshape(1, H * W, 1).to(mask_token.dtype)


# ------------------------------------------------------------------
# EditingSession: immutable state produced by Phase A (inversion)
# ------------------------------------------------------------------

@dataclass
class EditingSession:
    """Holds everything produced by Phase A that downstream phases need."""
    pipe: Any
    wrapper: Any
    model_type: str
    edit_init_latent: torch.Tensor      # [2, C, H_tok, W_tok]
    token_hw: Tuple[int, int]
    inv_trajectory: List[torch.Tensor] = field(default_factory=list)


# ------------------------------------------------------------------
# Phase A: encode + invert + pack
# ------------------------------------------------------------------

@torch.no_grad()
def prepare_session(
    wrapper: Any,
    model_type: str,
    image: Union[str, Image.Image],
    mask_np: np.ndarray,
    config: Dict[str, Any],
    blackout_unknown: bool = True,
    noise_unknown: bool = True,
) -> Tuple[EditingSession, torch.Tensor, torch.Tensor]:
    """Run Phase A and return (session, z_packed [2,N,C], mask_token [1,N,1]).

    Parameters
    ----------
    wrapper : Flux2UniEditFlowPipeline | FluxUniEditFlowPipeline
    model_type : ``"klein"`` or ``"flux1"``
    image : source image (path or PIL)
    mask_np : ``[1,1,H,W]`` float mask (1 = unknown)
    config : run config dict
    blackout_unknown : zero-out unknown pixels before VAE encode
    noise_unknown : replace mask-region latent tokens with noise after
        inversion.  Set to ``False`` for geometric modes where the object
        latent must be preserved for extraction and transformation.
    """
    pipe = wrapper.pipe
    T = int(config.get("T", 50))
    observed_reinject = config.get("observed_reinject", False)

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if blackout_unknown:
        mask_pil = Image.fromarray(
            (mask_np.squeeze() * 255).astype(np.uint8), mode="L"
        )
        bg = Image.new("RGB", image.size, (0, 0, 0))
        image = Image.composite(bg, image.convert("RGB"), mask_pil.resize(image.size))

    image_latent, _latent_ids = wrapper.image2latent(image)

    inv_trajectory: List[torch.Tensor] = []

    def _inv_callback(pipe_self, step, timestep, callback_kwargs):
        z = pipe_self.scheduler.sample
        if z is not None:
            inv_trajectory.append(z.clone().to(dtype))
        return callback_kwargs

    wrapper.invert_scheduler.set_hyperparameters(alpha=1.0)
    pipe.scheduler = wrapper.invert_scheduler
    invert_noise_latent = pipe(
        prompt="",
        num_inference_steps=T,
        guidance_scale=1.0,
        latents=image_latent.to(dtype),
        output_type="latent",
        height=512,
        width=512,
        callback_on_step_end=_inv_callback if observed_reinject else None,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images

    if model_type == "klein":
        invert_noise_latent = wrapper._patchify_and_bn_normalize(invert_noise_latent)
    else:
        invert_noise_latent = pipe._patchify_latents(invert_noise_latent)

    edit_init_latent = torch.cat([invert_noise_latent, invert_noise_latent], dim=0)
    _, _C, tok_h, tok_w = edit_init_latent.shape
    token_hw = (tok_h, tok_w)

    z_packed = pipe._pack_latents(edit_init_latent)  # [2, N, C]

    mask_token = mask_to_token_space(
        mask_np, token_hw, batch_size=1,
        device=z_packed.device, dtype=z_packed.dtype,
    )

    if noise_unknown:
        eps = torch.randn_like(z_packed[:1])
        m = mask_token
        z_packed = (1.0 - m) * z_packed + m * torch.cat([eps, eps], dim=0)

    session = EditingSession(
        pipe=pipe,
        wrapper=wrapper,
        model_type=model_type,
        edit_init_latent=edit_init_latent,
        token_hw=token_hw,
        inv_trajectory=inv_trajectory,
    )
    return session, z_packed, mask_token


# ------------------------------------------------------------------
# Phase B: warm start (thin wrapper for symmetry; heavy lifting in
#           edit.warm_start which already exists)
# ------------------------------------------------------------------

@torch.no_grad()
def run_warm_start(
    session: EditingSession,
    z_packed: torch.Tensor,
    mask_token: torch.Tensor,
    prompt_src: str,
    prompt_tgt: str,
    config: Dict[str, Any],
    debug_dir: Optional[str] = None,
) -> torch.Tensor:
    """Phase B wrapper. Returns updated ``z_packed``."""
    pipe = session.pipe
    T = int(config.get("T", 50))
    K = int(config.get("K", 10))
    warm_method = config.get("warm_method", "ot_harmonic")

    if K <= 0 or warm_method == "none":
        return z_packed

    warm_layers: List[Tuple[str, int]] = []
    for entry in config.get("warm_layers", [["double", 2], ["double", 4]]):
        warm_layers.append((str(entry[0]), int(entry[1])))

    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt_src, prompt_tgt],
        device=z_packed.device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    latent_image_ids = pipe._prepare_latent_ids(
        session.edit_init_latent
    ).to(z_packed.device)

    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    N = z_packed.shape[1]
    temp_sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    sigmas_np = np.linspace(1.0, 1.0 / T, T)
    mu = _compute_empirical_mu(image_seq_len=N, num_steps=T)
    temp_sched.set_timesteps(T, device=z_packed.device, sigmas=sigmas_np, mu=mu)
    sigmas_warm = temp_sched.sigmas[: K + 1].clone()

    try:
        from .warm_start import warm_start_loop
    except ImportError:
        from backbone.edit.warm_start import warm_start_loop
    config_with_hw = {**config, "token_hw": session.token_hw}
    return warm_start_loop(
        pipe=pipe,
        z_t=z_packed,
        mask_token=mask_token,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        latent_image_ids=latent_image_ids,
        sigmas_warm=sigmas_warm,
        warm_layers=warm_layers,
        config=config_with_hw,
        warm_method=warm_method,
        debug_dir=debug_dir,
    )


# ------------------------------------------------------------------
# Phase C: UniEdit-Flow editing (two variants)
# ------------------------------------------------------------------

@torch.no_grad()
def run_uniedit_phase(
    session: EditingSession,
    z_packed: torch.Tensor,
    mask_np: np.ndarray,
    prompt_src: str,
    prompt_tgt: str,
    config: Dict[str, Any],
    mask_token_override: Optional[torch.Tensor] = None,
    output_dir: str = "outputs",
) -> Tuple[torch.Tensor, Image.Image]:
    """Phase C: run UniEdit-Flow editing, return ``(z_packed_out, pil_image)``.

    Chooses reinject path when ``config["observed_reinject"]`` is truthy
    and ``session.inv_trajectory`` is non-empty.

    Parameters
    ----------
    mask_token_override : if given, used as the reinject mask instead of
        building one from *mask_np*.  Useful when the geometric pipeline
        has already built a composite reinject mask.
    """
    T = int(config.get("T", 50))
    K = int(config.get("K", 0))
    omega = float(config.get("omega", 5.0))
    alpha_edit = float(config.get("alpha_edit", 0.8))
    skip_alpha = (T - K) / T if K > 0 else alpha_edit
    debug = config.get("debug", False)
    observed_reinject = config.get("observed_reinject", False) and len(session.inv_trajectory) > 0
    reinject_mask_dilate = int(config.get("reinject_mask_dilate", 0))

    tok_h, tok_w = session.token_hw

    if observed_reinject:
        mask_token = mask_token_override if mask_token_override is not None else \
            mask_to_token_space(mask_np, session.token_hw, 1, z_packed.device, z_packed.dtype)
        z_out, img = _edit_with_reinject(
            session=session,
            z_packed=z_packed,
            mask_token=mask_token,
            mask_np=mask_np,
            prompt_src=prompt_src,
            prompt_tgt=prompt_tgt,
            T=T, K=K, omega=omega,
            skip_alpha=skip_alpha,
            reinject_mask_dilate=reinject_mask_dilate,
            debug=debug,
            output_dir=output_dir,
        )
    else:
        z_out, img = _edit_via_pipe(
            session=session,
            z_packed=z_packed,
            mask_np=mask_np,
            prompt_src=prompt_src,
            prompt_tgt=prompt_tgt,
            T=T, omega=omega,
            skip_alpha=skip_alpha,
            debug=debug,
            output_dir=output_dir,
        )
    return z_out, img


def _edit_via_pipe(
    session: EditingSession,
    z_packed: torch.Tensor,
    mask_np: np.ndarray,
    prompt_src: str,
    prompt_tgt: str,
    T: int,
    omega: float,
    skip_alpha: float,
    debug: bool,
    output_dir: str,
) -> Tuple[torch.Tensor, Image.Image]:
    pipe = session.pipe
    tok_h, tok_w = session.token_hw
    C_lat = session.edit_init_latent.shape[1]
    z_4d = z_packed.permute(0, 2, 1).reshape(2, C_lat, tok_h, tok_w)

    edit_sched = session.wrapper.edit_scheduler
    edit_sched.set_hyperparameters(alpha=skip_alpha, omega=omega)
    edit_sched.set_debug_options(
        save_masks=debug, print_mask_stats=False,
        mask_save_every=5, mask_dir=os.path.join(output_dir, "edit_masks"),
    )
    edit_sched.set_mask_token_shape(tok_h, tok_w)

    from flux_pipeline import _manual_mask_to_token_space
    external_mask = _manual_mask_to_token_space(
        manual_mask=mask_np, latent_hw=(tok_h, tok_w),
        batch_size=1, device=z_4d.device, dtype=z_4d.dtype,
    )
    edit_sched.set_external_guidance_mask(external_mask)

    pipe.scheduler = edit_sched
    img = pipe(
        prompt=[prompt_src, prompt_tgt],
        num_inference_steps=T, guidance_scale=1.0,
        latents=z_4d.to(dtype), max_sequence_length=512,
        height=512, width=512,
    ).images[0]
    return z_packed, img


def _edit_with_reinject(
    session: EditingSession,
    z_packed: torch.Tensor,
    mask_token: torch.Tensor,
    mask_np: np.ndarray,
    prompt_src: str,
    prompt_tgt: str,
    T: int,
    K: int,
    omega: float,
    skip_alpha: float,
    reinject_mask_dilate: int,
    debug: bool,
    output_dir: str,
) -> Tuple[torch.Tensor, Image.Image]:
    pipe = session.pipe
    tok_h, tok_w = session.token_hw
    N = z_packed.shape[1]

    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt_src, prompt_tgt],
        device=z_packed.device, num_images_per_prompt=1,
        max_sequence_length=512,
    )
    latent_image_ids = pipe._prepare_latent_ids(
        session.edit_init_latent
    ).to(z_packed.device)

    edit_sched = session.wrapper.edit_scheduler
    edit_sched.set_hyperparameters(alpha=skip_alpha, omega=omega)
    edit_sched.set_debug_options(
        save_masks=debug, print_mask_stats=False,
        mask_save_every=5, mask_dir=os.path.join(output_dir, "edit_masks"),
    )
    edit_sched.set_mask_token_shape(tok_h, tok_w)

    from flux_pipeline import _manual_mask_to_token_space
    external_mask = _manual_mask_to_token_space(
        manual_mask=mask_np, latent_hw=(tok_h, tok_w),
        batch_size=1, device=z_packed.device, dtype=z_packed.dtype,
    )
    edit_sched.set_external_guidance_mask(external_mask)

    sigmas_np = np.linspace(1.0, 1.0 / T, T)
    mu = _compute_empirical_mu(image_seq_len=N, num_steps=T)
    edit_sched.set_timesteps(T, device=z_packed.device, sigmas=sigmas_np, mu=mu)
    timesteps = edit_sched.timesteps
    edit_sched._step_index = 0
    edit_sched._begin_index = 0

    m_reinject = dilate_token_mask(mask_token, session.token_hw, reinject_mask_dilate)
    inv_trajectory = session.inv_trajectory

    z = z_packed
    for j, t in enumerate(timesteps):
        reinject_idx = max(0, min(T - K - j, len(inv_trajectory) - 1))
        z_inv_batch = inv_trajectory[reinject_idx].expand(2, -1, -1)
        z = reinject_observed_tokens(z, z_inv_batch, m_reinject)

        timestep = t.expand(z.shape[0]).to(z.dtype)
        noise_pred = pipe.transformer(
            hidden_states=z.to(pipe.transformer.dtype),
            timestep=timestep / 1000.0,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0][:, :N, :]

        z = edit_sched.step(noise_pred, t, z, return_dict=False)[0]

    if len(inv_trajectory) > 0:
        z_clean = inv_trajectory[0].expand(2, -1, -1)
        z = reinject_observed_tokens(z, z_clean, m_reinject)

    img = decode_latent(session, z)
    return z, img


# ------------------------------------------------------------------
# Phase D: decode packed latent → PIL image
# ------------------------------------------------------------------

@torch.no_grad()
def decode_latent(session: EditingSession, z_packed: torch.Tensor) -> Image.Image:
    """Decode a ``[2, N, C]`` packed latent (takes source branch row 0)."""
    pipe = session.pipe
    latent_image_ids = pipe._prepare_latent_ids(
        session.edit_init_latent
    ).to(z_packed.device)

    z_src = z_packed[:1]
    ids_src = latent_image_ids[:1]
    z_4d = pipe._unpack_latents_with_ids(z_src, ids_src)

    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(z_4d.device, z_4d.dtype)
    bn_std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    ).to(z_4d.device, z_4d.dtype)
    z_4d = z_4d * bn_std + bn_mean

    z_4d = pipe._unpatchify_latents(z_4d)
    decoded = pipe.vae.decode(z_4d, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]
