"""BCOTHVEPipeline -- orchestrator for BCOT-HVE inpainting / outpainting.

Wraps the existing Flux2UniEditFlowPipeline (or FluxUniEditFlowPipeline)
and inserts a K-step warm-start phase before the standard UniEdit-Flow
editing loop.  No existing scheduler or pipeline code is modified.

Observed-region reinjection: at every post-warm-start denoising step,
the observed (known) region latent is hard-replaced with the
corresponding tokens from the stored inversion trajectory, preventing
colorimetric drift.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .warm_start import warm_start_loop
    from .observed_reinject import reinject_observed_tokens
    from ..utils.mask_ops import (
        infer_token_hw,
        mask_to_token_space,
        resize_mask_to_latent,
        mask_2d_from_token_mask,
        extract_boundary_band,
        reinject_mask_token_expanded_unknown,
        dilate_token_mask,
    )
except ImportError:
    from backbone.edit.warm_start import warm_start_loop
    from backbone.edit.observed_reinject import reinject_observed_tokens
    from backbone.utils.mask_ops import (
        infer_token_hw,
        mask_to_token_space,
        resize_mask_to_latent,
        mask_2d_from_token_mask,
        extract_boundary_band,
        reinject_mask_token_expanded_unknown,
        dilate_token_mask,
    )

device = "cuda"
dtype = torch.bfloat16


def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Reproduce the mu computation from Flux2KleinPipeline."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


class BCOTHVEPipeline:
    """End-to-end BCOT-HVE inpainting / outpainting controller.

    Parameters
    ----------
    offload : bool
        If True, enable CPU offload for the underlying model.
    model : str
        ``"klein"`` for Flux2-Klein-4B or ``"flux1"`` for Flux.1-dev.
    device : str
        Torch device for model execution, e.g. ``"cuda"`` or ``"cuda:1"``.
    """

    def __init__(self, offload: bool = False, model: str = "klein", device: str = "cuda"):
        self.device = str(device)
        if model == "klein":
            try:
                from ..flux_pipeline import Flux2UniEditFlowPipeline
            except ImportError:
                from backbone.flux_pipeline import Flux2UniEditFlowPipeline
            self.wrapper = Flux2UniEditFlowPipeline(offload=offload, device=self.device)
        else:
            try:
                from ..flux_pipeline import FluxUniEditFlowPipeline
            except ImportError:
                from backbone.flux_pipeline import FluxUniEditFlowPipeline
            self.wrapper = FluxUniEditFlowPipeline(offload=offload, device=self.device)
        self.model_type = model

    @torch.no_grad()
    def run(
        self,
        image: Union[str, Image.Image],
        mask: Union[str, np.ndarray, torch.Tensor],
        prompt_src: str,
        prompt_tgt: str,
        config: Dict[str, Any],
        output_dir: Optional[str] = None,
        blackout_unknown: bool = True,
    ) -> Image.Image:
        """Execute the full four-phase BCOT-HVE pipeline.

        Returns the final decoded PIL image.
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        T = int(config.get("T", 50))
        K = int(config.get("K", 10))
        warm_method = config.get("warm_method", "ot_harmonic")
        omega = float(config.get("omega", 5.0))
        alpha_edit = float(config.get("alpha_edit", 0.8))
        skip_alpha = (T - K) / T if K > 0 else alpha_edit
        debug = config.get("debug", False)
        observed_reinject = config.get("observed_reinject", False)
        vae_observed_color_fix = bool(config.get("vae_observed_color_fix", True))

        warm_layers: List[Tuple[str, int]] = []
        for entry in config.get("warm_layers", [["double", 2], ["double", 4]]):
            warm_layers.append((str(entry[0]), int(entry[1])))

        pipe = self.wrapper.pipe  # the raw Flux2KleinPipeline / FluxPipeline

        # ------------------------------------------------------------------
        # Phase A: setup -- encode, invert, initialise noisy latent
        # ------------------------------------------------------------------
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(mask, str):
            mask_img = Image.open(mask).convert("L")
            mask_np = np.array(mask_img.resize((512, 512))) / 255.0
            mask_np = mask_np.astype(np.float32).reshape(1, 1, 512, 512)
        elif isinstance(mask, np.ndarray):
            mask_np = mask
        else:
            mask_np = mask.cpu().numpy()

        # Black out unknown region to prevent info leakage if blackout_unknown is True
        if blackout_unknown:
            mask_pil = Image.fromarray((mask_np.squeeze() * 255).astype(np.uint8), mode="L")
            bg = Image.new("RGB", image.size, (0, 0, 0))
            image = Image.composite(bg, image.convert("RGB"), mask_pil.resize(image.size))

        # VAE encode
        image_latent, latent_ids = self.wrapper.image2latent(image)

        # -- Inversion with optional trajectory capture ----
        inv_trajectory: List[torch.Tensor] = []

        def _inv_callback(pipe_self, step, timestep, callback_kwargs):
            """Store scheduler.sample (true state) at each inversion step."""
            z = pipe_self.scheduler.sample
            if z is not None:
                inv_trajectory.append(z.clone().to(dtype))
            return callback_kwargs

        self.wrapper.invert_scheduler.set_hyperparameters(alpha=1.0)
        pipe.scheduler = self.wrapper.invert_scheduler
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

        if self.model_type == "klein":
            invert_noise_latent = self.wrapper._patchify_and_bn_normalize(invert_noise_latent)
        else:
            invert_noise_latent = pipe._patchify_latents(invert_noise_latent)

        # Build batch-doubled 4-D latent [2, C, H_tok, W_tok]
        edit_init_latent = torch.cat([invert_noise_latent, invert_noise_latent], dim=0)
        _, C_lat, tok_h, tok_w = edit_init_latent.shape
        N = tok_h * tok_w
        token_hw = (tok_h, tok_w)

        # Pack to [2, N, C] (transformer token format)
        z_packed = pipe._pack_latents(edit_init_latent)  # [2, N, C]

        # Build token-space mask [1, N, 1]
        mask_token = mask_to_token_space(
            mask_np, token_hw, batch_size=1, device=z_packed.device, dtype=z_packed.dtype,
        )

        reinject_expand_px = int(config.get("reinject_unknown_expand_px", 0))
        reinject_mask_dilate = int(config.get("reinject_mask_dilate", 0))
        mask_token_reinject = reinject_mask_token_expanded_unknown(
            mask_np,
            token_hw,
            batch_size=1,
            device=z_packed.device,
            dtype=z_packed.dtype,
            expand_unknown_pixels=reinject_expand_px,
        )
        if reinject_mask_dilate > 0:
            mask_token_reinject = dilate_token_mask(
                mask_token_reinject, token_hw, reinject_mask_dilate
            )

        # Replace unknown region with Gaussian noise
        eps = torch.randn_like(z_packed[:1])  # [1, N, C]
        m = mask_token  # [1, N, 1]
        z_packed = (1.0 - m) * z_packed + m * torch.cat([eps, eps], dim=0)

        if debug and output_dir is not None:
            torch.save(z_packed.cpu(), os.path.join(output_dir, "z_before_warmstart.pt"))

        # ------------------------------------------------------------------
        # Phase B: K-step warm start (skip if warm_method == "none" and K==0)
        # ------------------------------------------------------------------
        if K > 0 and warm_method != "none":
            # Encode prompts
            prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=[prompt_src, prompt_tgt],
                device=z_packed.device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )

            # Prepare latent IDs from the 4-D latent
            latent_image_ids = pipe._prepare_latent_ids(edit_init_latent).to(z_packed.device)

            # Build full sigma schedule (T steps, alpha=1)
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
            temp_sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            sigmas_np = np.linspace(1.0, 1.0 / T, T)
            mu = _compute_empirical_mu(image_seq_len=N, num_steps=T)
            temp_sched.set_timesteps(T, device=z_packed.device, sigmas=sigmas_np, mu=mu)
            all_sigmas = temp_sched.sigmas  # [T+1] (includes trailing 0)
            sigmas_warm = all_sigmas[: K + 1].clone()

            config_with_hw = {**config, "token_hw": token_hw}
            debug_warmdir = os.path.join(output_dir, "warm_debug") if (debug and output_dir is not None) else None

            z_packed = warm_start_loop(
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
                debug_dir=debug_warmdir,
            )

        if debug and output_dir is not None:
            torch.save(z_packed.cpu(), os.path.join(output_dir, "z_after_warmstart.pt"))

        # ------------------------------------------------------------------
        # Phase C: UniEdit-Flow for the remaining T-K steps
        #   When observed_reinject=True, a manual loop replaces the
        #   black-box pipe() call so we can inject inversion-trajectory
        #   tokens into the observed region before each velocity prediction.
        # ------------------------------------------------------------------
        if observed_reinject:
            result_image = self._edit_with_reinject(
                pipe=pipe,
                z_packed=z_packed,
                reinject_mask_token=mask_token_reinject,
                mask_np=mask_np,
                inv_trajectory=inv_trajectory,
                prompt_src=prompt_src,
                prompt_tgt=prompt_tgt,
                edit_init_latent=edit_init_latent,
                token_hw=token_hw,
                T=T,
                K=K,
                omega=omega,
                skip_alpha=skip_alpha,
                debug=debug,
                output_dir=output_dir,
                reference_image=image,
                vae_observed_color_fix=vae_observed_color_fix,
                color_fix_config=config,
            )
        else:
            result_image = self._edit_via_pipe(
                pipe=pipe,
                z_packed=z_packed,
                mask_np=mask_np,
                edit_init_latent=edit_init_latent,
                token_hw=token_hw,
                T=T,
                omega=omega,
                skip_alpha=skip_alpha,
                prompt_src=prompt_src,
                prompt_tgt=prompt_tgt,
                debug=debug,
                output_dir=output_dir,
                reference_image=image,
                vae_observed_color_fix=vae_observed_color_fix,
                color_fix_config=config,
            )

        # ------------------------------------------------------------------
        # Phase D: save outputs
        # ------------------------------------------------------------------
        if output_dir is not None:
            result_image.save(os.path.join(output_dir, "result.png"))

        if debug and output_dir is not None:
            self._save_debug_outputs(image, result_image, mask_np, token_hw, output_dir)

        return result_image

    # ------------------------------------------------------------------
    # Phase C helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vae_affine_color_fix_decoded(
        decoded: torch.Tensor,
        ref_tensor: torch.Tensor,
        mask_unknown_hw: torch.Tensor,
        min_pixels: int = 64,
        scale_lo: float = 0.5,
        scale_hi: float = 2.0,
    ) -> torch.Tensor:
        """Per-channel affine match ``decoded ≈ scale * decoded + shift`` to ``ref`` on observed pixels.

        ``decoded`` and ``ref_tensor`` are ``[1, 3, H, W]`` in the same value range (e.g. ``[-1, 1]``).
        ``mask_unknown_hw`` is ``[1, 1, H, W]`` with **1 = unknown**, **0 = observed**; the fit uses
        **observed** pixels only.
        """
        if decoded.shape != ref_tensor.shape:
            raise ValueError(f"decoded/ref shape mismatch: {tuple(decoded.shape)} vs {tuple(ref_tensor.shape)}")
        H, W = decoded.shape[-2:]
        if mask_unknown_hw.shape[-2:] != (H, W):
            mask_unknown_hw = F.interpolate(
                mask_unknown_hw.float(), size=(H, W), mode="bilinear", align_corners=False
            )
        known = (mask_unknown_hw < 0.5).float()  # [1,1,H,W]
        out = decoded.clone()
        for c in range(3):
            d = decoded[0, c].reshape(-1)
            r = ref_tensor[0, c].reshape(-1)
            k = known.reshape(-1) > 0.5
            n = int(k.sum().item())
            if n < min_pixels:
                continue
            dv = d[k].float()
            rv = r[k].float()
            mean_d = dv.mean()
            mean_r = rv.mean()
            var_d = ((dv - mean_d) ** 2).mean().clamp(min=1e-8)
            cov = ((dv - mean_d) * (rv - mean_r)).mean()
            scale = (cov / var_d).clamp(scale_lo, scale_hi).to(decoded.dtype)
            shift = (mean_r - scale * mean_d).to(decoded.dtype)
            out[0, c] = (decoded[0, c] * scale + shift).clamp(-1.0, 1.0)
        return out

    def _vae_color_fix_prepare_reference_tensor(
        self,
        pipe,
        reference_image: Image.Image,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ref_t = pipe.image_processor.preprocess(
            reference_image.convert("RGB"), height=height, width=width
        )
        ref_t = ref_t.to(device=device, dtype=dtype)
        if ref_t.ndim == 3:
            ref_t = ref_t.unsqueeze(0)
        return ref_t

    @torch.no_grad()
    def _edit_via_pipe(
        self,
        pipe,
        z_packed: torch.Tensor,
        mask_np: np.ndarray,
        edit_init_latent: torch.Tensor,
        token_hw: Tuple[int, int],
        T: int,
        omega: float,
        skip_alpha: float,
        prompt_src: str,
        prompt_tgt: str,
        debug: bool,
        output_dir: Optional[str] = None,
        reference_image: Optional[Image.Image] = None,
        vae_observed_color_fix: bool = True,
        color_fix_config: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        """Original Phase C: delegate to the pipeline black-box."""
        tok_h, tok_w = token_hw
        C_lat = edit_init_latent.shape[1]
        z_4d = z_packed.permute(0, 2, 1).reshape(2, C_lat, tok_h, tok_w)

        self.wrapper.edit_scheduler.set_hyperparameters(alpha=skip_alpha, omega=omega)
        self.wrapper.edit_scheduler.set_debug_options(
            save_masks=debug and output_dir is not None,
            print_mask_stats=False,
            mask_save_every=5,
            mask_dir=os.path.join(output_dir, "edit_masks") if (output_dir is not None) else None,
        )
        self.wrapper.edit_scheduler.set_mask_token_shape(tok_h, tok_w)

        try:
            from ..flux_pipeline import _manual_mask_to_token_space
        except ImportError:
            from backbone.flux_pipeline import _manual_mask_to_token_space
        external_mask = _manual_mask_to_token_space(
            manual_mask=mask_np,
            latent_hw=(tok_h, tok_w),
            batch_size=1,
            device=z_4d.device,
            dtype=z_4d.dtype,
        )
        self.wrapper.edit_scheduler.set_external_guidance_mask(external_mask)

        pipe.scheduler = self.wrapper.edit_scheduler
        result_pil = pipe(
            prompt=[prompt_src, prompt_tgt],
            num_inference_steps=T,
            guidance_scale=1.0,
            latents=z_4d.to(dtype),
            max_sequence_length=512,
            height=512,
            width=512,
        ).images[0]
        if vae_observed_color_fix and reference_image is not None:
            cfg = color_fix_config or {}
            result_pil = self._vae_affine_color_fix_pil(
                result_pil,
                reference_image,
                mask_np,
                min_pixels=int(cfg.get("vae_color_fix_min_pixels", 64)),
                scale_lo=float(cfg.get("vae_color_fix_scale_lo", 0.5)),
                scale_hi=float(cfg.get("vae_color_fix_scale_hi", 2.0)),
            )
        return result_pil

    @torch.no_grad()
    def _edit_with_reinject(
        self,
        pipe,
        z_packed: torch.Tensor,
        reinject_mask_token: torch.Tensor,
        mask_np: np.ndarray,
        inv_trajectory: List[torch.Tensor],
        prompt_src: str,
        prompt_tgt: str,
        edit_init_latent: torch.Tensor,
        token_hw: Tuple[int, int],
        T: int,
        K: int,
        omega: float,
        skip_alpha: float,
        debug: bool,
        output_dir: Optional[str],
        reference_image: Optional[Image.Image] = None,
        vae_observed_color_fix: bool = True,
        color_fix_config: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        """Phase C with observed-region reinjection from inversion trajectory.

        Replaces the black-box pipe() call with a manual denoising loop
        that reinjects observed-region tokens before each transformer call.

        ``reinject_mask_token`` may widen the unknown region versus the
        edit-phase mask (see ``reinject_unknown_expand_px``) so boundary
        tokens are not forced to inversion latents that saw composite bleed.
        """
        tok_h, tok_w = token_hw
        N = z_packed.shape[1]

        # -- Prepare prompt embeddings & IDs --
        print("prompt_src: ", prompt_src)
        print("prompt_tgt: ", prompt_tgt)
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=[prompt_src, prompt_tgt],
            device=z_packed.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        latent_image_ids = pipe._prepare_latent_ids(edit_init_latent).to(z_packed.device)

        # -- Configure edit scheduler --
        self.wrapper.edit_scheduler.set_hyperparameters(alpha=skip_alpha, omega=omega)
        self.wrapper.edit_scheduler.set_debug_options(
            save_masks=debug and output_dir is not None,
            print_mask_stats=False,
            mask_save_every=5,
            mask_dir=os.path.join(output_dir, "edit_masks") if (output_dir is not None) else None,
        )
        self.wrapper.edit_scheduler.set_mask_token_shape(tok_h, tok_w)

        try:
            from ..flux_pipeline import _manual_mask_to_token_space
        except ImportError:
            from backbone.flux_pipeline import _manual_mask_to_token_space
        external_mask = _manual_mask_to_token_space(
            manual_mask=mask_np,
            latent_hw=(tok_h, tok_w),
            batch_size=1,
            device=z_packed.device,
            dtype=z_packed.dtype,
        )
        self.wrapper.edit_scheduler.set_external_guidance_mask(external_mask)

        # -- Set up sigma schedule --
        edit_sched = self.wrapper.edit_scheduler
        sigmas_np = np.linspace(1.0, 1.0 / T, T)
        mu = _compute_empirical_mu(image_seq_len=N, num_steps=T)
        edit_sched.set_timesteps(T, device=z_packed.device, sigmas=sigmas_np, mu=mu)
        timesteps = edit_sched.timesteps
        num_edit_steps = len(timesteps)
        edit_sched._step_index = 0
        edit_sched._begin_index = 0

        if debug:
            print(
                f"[reinject] T={T} K={K} edit_steps={num_edit_steps} "
                f"inv_traj_len={len(inv_trajectory)} "
                f"sigma_range=[{float(edit_sched.sigmas[0]):.4f}, "
                f"{float(edit_sched.sigmas[-1]):.4f}]"
            )

        m = reinject_mask_token  # [1, N, 1], 1=unknown 0=observed (may widen hole vs. edit mask)
        z = z_packed    # [2, N, C]

        # -- Manual denoising loop with reinjection --
        for j, t in enumerate(timesteps):
            # Index into inv_trajectory: map edit step j → inversion step
            reinject_idx = T - K - j
            reinject_idx = max(0, min(reinject_idx, len(inv_trajectory) - 1))
            z_inv = inv_trajectory[reinject_idx]  # [1, N, C]
            z_inv_batch = z_inv.expand(2, -1, -1)  # [2, N, C]

            # Reinject observed tokens from inversion trajectory
            z = reinject_observed_tokens(z, z_inv_batch, m)

            if debug and j % 10 == 0:
                obs_diff = (z[:1] - z_inv)[:, m[0, :, 0] < 0.5, :].abs().mean()
                print(
                    f"[reinject] step {j}/{num_edit_steps} "
                    f"reinject_idx={reinject_idx} "
                    f"obs_region_diff={obs_diff:.6f}"
                )

            # Transformer forward
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
            )[0]
            noise_pred = noise_pred[:, :N, :]

            # Scheduler step (UniEditEulerScheduler handles velocity fusion)
            z = edit_sched.step(noise_pred, t, z, return_dict=False)[0]

        # Final reinjection: restore observed region to clean-image tokens
        # (inv_trajectory[0] = z at sigma=0, the original image latent)
        if len(inv_trajectory) > 0:
            z_clean = inv_trajectory[0].expand(2, -1, -1)
            z = reinject_observed_tokens(z, z_clean, m)

        # -- Decode --
        z_src = z[:1]  # [1, N, C]
        ids_src = latent_image_ids[:1]  # [1, N, 4]
        z_4d = pipe._unpack_latents_with_ids(z_src, ids_src)  # [1, C_patch, H, W]

        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(z_4d.device, z_4d.dtype)
        bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
        ).to(z_4d.device, z_4d.dtype)
        z_4d = z_4d * bn_std + bn_mean

        z_4d = pipe._unpatchify_latents(z_4d)
        decoded = pipe.vae.decode(z_4d, return_dict=False)[0]
        if vae_observed_color_fix and reference_image is not None:
            cfg = color_fix_config or {}
            H, W = decoded.shape[-2], decoded.shape[-1]
            m_t = torch.from_numpy(mask_np.astype(np.float32)).to(decoded.device)
            if m_t.ndim == 2:
                m_t = m_t.reshape(1, 1, *m_t.shape)
            elif m_t.ndim == 3:
                m_t = m_t.unsqueeze(0)
            elif m_t.ndim == 4:
                pass
            else:
                raise ValueError(f"mask_np must be 2D–4D after numpy conversion, got {m_t.shape}")
            ref_t = self._vae_color_fix_prepare_reference_tensor(
                pipe, reference_image, H, W, decoded.device, decoded.dtype,
            )
            decoded = self._vae_affine_color_fix_decoded(
                decoded,
                ref_t,
                m_t,
                min_pixels=int(cfg.get("vae_color_fix_min_pixels", 64)),
                scale_lo=float(cfg.get("vae_color_fix_scale_lo", 0.5)),
                scale_hi=float(cfg.get("vae_color_fix_scale_hi", 2.0)),
            )
        result_image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

        return result_image

    @staticmethod
    def _vae_affine_color_fix_pil(
        result: Image.Image,
        reference: Image.Image,
        mask_np: np.ndarray,
        min_pixels: int = 64,
        scale_lo: float = 0.5,
        scale_hi: float = 2.0,
    ) -> Image.Image:
        """Same affine model as ``_vae_affine_color_fix_decoded`` but on linear RGB ``[0, 1]`` PIL images."""
        res = np.array(result.convert("RGB"), dtype=np.float32) / 255.0
        ref = np.array(reference.convert("RGB").resize(result.size), dtype=np.float32) / 255.0
        m = mask_np.squeeze()
        if m.shape != res.shape[:2]:
            m = (
                np.array(
                    Image.fromarray((m * 255).astype(np.uint8), mode="L").resize(
                        (res.shape[1], res.shape[0])
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
        known = m < 0.5
        if int(known.sum()) < min_pixels:
            return result
        out = res.copy()
        for c in range(3):
            d = res[:, :, c][known].reshape(-1)
            r = ref[:, :, c][known].reshape(-1)
            mean_d = float(d.mean())
            mean_r = float(r.mean())
            var_d = float(np.mean((d - mean_d) ** 2)) + 1e-8
            cov = float(np.mean((d - mean_d) * (r - mean_r)))
            scale = float(np.clip(cov / var_d, scale_lo, scale_hi))
            shift = mean_r - scale * mean_d
            out[:, :, c] = np.clip(scale * res[:, :, c] + shift, 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")

    def compute_metrics(
        self,
        original: Image.Image,
        result: Image.Image,
        mask_np: np.ndarray,
    ) -> Dict[str, float]:
        """Compute preservation error and boundary seam score.

        Parameters
        ----------
        original : PIL image before masking.
        result : decoded output image.
        mask_np : ``[1, 1, H, W]`` mask array (1 = unknown).

        Returns
        -------
        dict with ``preservation_error`` and ``boundary_seam_score``.
        """
        orig_arr = np.array(original.resize((512, 512))).astype(np.float32) / 255.0
        res_arr = np.array(result.resize((512, 512))).astype(np.float32) / 255.0
        m = mask_np.squeeze()  # [H, W]
        if m.shape != orig_arr.shape[:2]:
            from PIL import Image as _Img
            m = np.array(
                _Img.fromarray((m * 255).astype(np.uint8)).resize(
                    (orig_arr.shape[1], orig_arr.shape[0])
                )
            ).astype(np.float32) / 255.0

        known = m < 0.5
        if known.sum() > 0:
            pres_err = float(np.mean((orig_arr[known] - res_arr[known]) ** 2))
        else:
            pres_err = 0.0

        # Boundary seam score: gradient magnitude in a thin band around the mask edge
        gray = np.mean(res_arr, axis=-1)
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        m_t = torch.from_numpy(m).float()
        band = extract_boundary_band(m_t, width=3).numpy() > 0.5
        dilated_mask = F.max_pool2d(
            torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0),
            kernel_size=7, stride=1, padding=3,
        ).squeeze().numpy() > 0.5
        inner_band = dilated_mask & (m > 0.5)
        full_band = band | inner_band

        if full_band.sum() > 0:
            seam_score = float(np.mean(grad_mag[full_band]))
        else:
            seam_score = 0.0

        return {"preservation_error": pres_err, "boundary_seam_score": seam_score}

    def _save_debug_outputs(
        self,
        original: Image.Image,
        result: Image.Image,
        mask_np: np.ndarray,
        token_hw: Tuple[int, int],
        output_dir: Optional[str] = None,
    ) -> None:
        """Save debug outputs to the output directory."""
        if output_dir is not None:
            original.save(os.path.join(output_dir, "original_masked.png"))
            m_vis = (mask_np.squeeze() * 255).astype(np.uint8)
            Image.fromarray(m_vis, mode="L").save(os.path.join(output_dir, "mask.png"))

            metrics = self.compute_metrics(original, result, mask_np)
            with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v:.6f}\n")
            print(f"[BCOT-HVE] Metrics: {metrics}")
