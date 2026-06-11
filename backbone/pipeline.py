"""WonderWorld backbone pipeline class for inpainting / outpainting with boundary-constrained distribution matching (BCDM)"""

import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline, FluxPipeline

# Import of our own UniEdit-Flow schedulers
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
try:
    from .uniedit_flow_schedulers.UniInvEulerScheduler import UniInvEulerScheduler
    from .uniedit_flow_schedulers.UniEditEulerScheduler import UniEditEulerScheduler
except ImportError:
    try:
        from backbone.uniedit_flow_schedulers.UniInvEulerScheduler import UniInvEulerScheduler
        from backbone.uniedit_flow_schedulers.UniEditEulerScheduler import UniEditEulerScheduler
    except ImportError:
        from uniedit_flow_schedulers.UniInvEulerScheduler import UniInvEulerScheduler
        from uniedit_flow_schedulers.UniEditEulerScheduler import UniEditEulerScheduler

# Miscellaneous imports
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# Utils
try:
    from utils.mask_ops import (
        mask_image_to_numpy,
        mask_to_token_space,
        resize_mask_to_latent,
        mask_2d_from_token_mask,
        extract_boundary_band,
        reinject_mask_token_expanded_unknown,
        dilate_token_mask,
    )
except ImportError:
    from backbone.utils.mask_ops import (
        mask_image_to_numpy,
        mask_to_token_space,
        resize_mask_to_latent,
        mask_2d_from_token_mask,
        extract_boundary_band,
        reinject_mask_token_expanded_unknown,
        dilate_token_mask,
    )

# Optimal transport and Poisson interpolation for BCDM warm-start
try:
    from edit.optimal_transport import (
        build_transport_cost,
        sinkhorn_transport,
    )
    from edit.poisson_interpolation import poisson_interp
    from utils.mask_ops import (
        build_token_grid,
        mask_2d_from_token_mask,
    )
except ImportError:
    from backbone.edit.optimal_transport import (
        build_transport_cost,
        sinkhorn_transport,
    )
    from backbone.edit.poisson_interpolation import poisson_interp
    from backbone.utils.mask_ops import (
        build_token_grid,
        mask_2d_from_token_mask,
    )

# Geometry edit utilities
try:
    from geometry.spec import EditType, GeometrySpec
    from geometry.utils import (
        build_boundary_blur_mask,
        build_inpainting_mask,
        geom_mask_to_np,
        inverse_affine_coeffs_for_pil,
        mask_tensor_to_pil_l,
    )
except ImportError:
    from backbone.geometry.spec import EditType, GeometrySpec
    from backbone.geometry.utils import (
        build_boundary_blur_mask,
        build_inpainting_mask,
        geom_mask_to_np,
        inverse_affine_coeffs_for_pil,
        mask_tensor_to_pil_l,
    )

dtype = torch.bfloat16

# Copied from diffusers.pipelines.flux2.pipeline_flux2.compute_empirical_mu
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

class BackbonePipeline:
    def __init__(self, offload=False, device: str = "cuda"):
        self.device = str(device)
        
        # FLUX.2 [klein] 4B as the base model
        model_path = "black-forest-labs/FLUX.2-klein-base-4B"

        self.pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype)

        # We utilise our existing UniEdit-Flow schedulers for inversion and editing.
        # Uni-Inv scheduler as-is for inversion
        self.invert_scheduler = UniInvEulerScheduler.from_config(self.pipe.scheduler.config)

        # Modified Uni-Edit scheduler for editing (with manual guidance mask)
        self.edit_scheduler = UniEditEulerScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        if offload:
            self.pipe.enable_model_cpu_offload(device=self.device)  # save VRAM; execute on the configured GPU
        else:
            self.pipe.to(self.device)

    # VAE encode image to latent.
    @torch.no_grad()
    def image_to_latent(self, image):
        if not hasattr(self.pipe.vae, "_hf_hook"):
            self.pipe.vae.to(self.device)
        vae_device = getattr(self.pipe, "_offload_device", next(self.pipe.vae.parameters()).device)
        image = self.pipe.image_processor.preprocess(image).type(dtype).to(vae_device)
        latents = self.pipe._encode_vae_image(image, generator=None)
        latent_ids = self.pipe._prepare_latent_ids(latents)
        return latents, latent_ids

    # Patchify and BN normalize latent according to FLUX.2 VAE's procedure
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._encode_vae_image
    def patchify_and_bn_normalize_latent(self, latents):
        latents = self.pipe._patchify_latents(latents)
        bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        latents = (latents - bn_mean) / bn_std
        return latents

    # VAE decode latent (z_0 from report) to image
    @torch.no_grad()
    def latent_to_image(self, z_0, latent_image_ids, vae_color_fix=True, unknown_mask=None, reference_image=None):
        z_src = z_0[:1]
        ids_src = latent_image_ids[:1]
        # Unpack latent with IDs
        z_unpacked = self.pipe._unpack_latents_with_ids(z_src, ids_src)

        # BN un-normalise latents
        bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(z_unpacked.device, z_unpacked.dtype)
        bn_std = torch.sqrt(self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps).to(z_unpacked.device, z_unpacked.dtype)
        z_unpacked = z_unpacked * bn_std + bn_mean

        # Unpatchify latents
        z_unpacked = self.pipe._unpatchify_latents(z_unpacked)

        # Decode latents to image
        decoded = self.pipe.vae.decode(z_unpacked, return_dict=False)[0]
        
        # VAE color fix for color drift
        if vae_color_fix and unknown_mask is not None and reference_image is not None:
            H, W = decoded.shape[-2], decoded.shape[-1]
            m_t = torch.from_numpy(unknown_mask.astype(np.float32)).to(decoded.device)
            ref_t = self.pipe.image_processor.preprocess(
                reference_image.convert("RGB"), height=512, width=512
            )
            ref_t = ref_t.to(self.device, dtype=z_0.dtype)
            if ref_t.ndim == 3:
                ref_t = ref_t.unsqueeze(0)

            decoded = vae_affine_color_fix_decoded(decoded, ref_t, m_t)

        return self.pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    # Invert image to noisy latent
    @torch.no_grad()
    def invert_image(
        self,
        image_latent, # VAE encoded latent
        latent_ids, # Latent IDs
        steps=50, # Number of inversion steps
        store_for_reinjection=True
    ):
        # Utilise Uni-Inv scheduler for inversion
        self.invert_scheduler.set_hyperparameters(alpha=1.0)

        # Store intermediate latents for reinjection of known region
        inversion_trajectory: List[torch.Tensor] = []
        def store_intermediate_latents(pipe_self, step, timestep, callback_kwargs):
            z = pipe_self.scheduler.sample
            if z is not None:
                inversion_trajectory.append(z.clone().to(dtype))
            return callback_kwargs
        
        # Invert
        self.pipe.scheduler = self.invert_scheduler
        invert_noise_latent = self.pipe(
            prompt="",
            num_inference_steps=steps,
            guidance_scale=1.0,
            latents=image_latent.to(dtype),
            output_type='latent',
            height=512,
            width=512,
            callback_on_step_end=store_intermediate_latents if store_for_reinjection else None,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images

        # Patchify and BN normalize latent
        invert_noise_latent = self.patchify_and_bn_normalize_latent(invert_noise_latent)

        # Return noisy latent and intermediate latents for reinjection (empty list if not storing)
        return invert_noise_latent, inversion_trajectory

    # Boundary-constrained distribution matching (BCDM)
    @torch.no_grad()
    def bcdm_loop(
        self,
        z_1, # Noisy latent (z_1 from report)
        unknown_token_mask, # Flattened token-space mask for unknown region
        unknown_token_mask_2d, # 2D token-space mask for unknown region
        token_hw, # Token grid dimensions
        prompt_embeds, # Source and target prompt embeddings
        text_ids, # Source and target token IDs
        latent_image_ids, # Source and target latent IDs
        # Warm start steps
        warm_start_steps=10,
        # Sigmas from Uni-Edit scheduler
        sigmas=None,
        # Alpha schedule
        alpha_start=0.95,
        alpha_end=0.8,
        warm_layers=[ # Stable Flow vital layers (from appendix)
            ["double", 0],
            ["double", 4],
            ["single", 4],
            ["single", 5],
            ["single", 15],
            ["single", 19]
        ]
    ):
        if warm_start_steps <= 0:
            return z_1

        N = z_1.shape[1]
        binarised_unknown_token_mask = unknown_token_mask[0, :, 0] > 0.5
        known_token_mask = ~binarised_unknown_token_mask

        # Build token grid coordinates for OT and poisson interpolation steps
        coords = build_token_grid(token_hw, device=z_1.device, dtype=torch.float32)
        unk_idx = binarised_unknown_token_mask.nonzero(as_tuple=False).squeeze(-1)
        kn_idx = known_token_mask.nonzero(as_tuple=False).squeeze(-1)
        pos_unk = coords[unk_idx]
        pos_kn = coords[kn_idx]
        num_txt = prompt_embeds.shape[1]

        # Initialise hook manager for OT step feature extraction for cost matrix calculation
        hook_mgr = BCDMHookManager(self.pipe.transformer, warm_layers, num_txt)
        hook_mgr.register()

        m = unknown_token_mask.to(z_1.device, dtype=z_1.dtype)

        # BCDM loop
        try:
            for k in tqdm(range(warm_start_steps), total=warm_start_steps, desc="BCDM warm start"):
                sigma = sigmas[k]
                sigma_next = sigmas[k + 1]
                dt = sigma_next - sigma  # negative (descending)
                timestep = (sigma * 1000.0).expand(z_1.shape[0]).to(dtype)
                
                if hook_mgr is not None:
                    hook_mgr.clear()

                v_pred = self.pipe.transformer(
                    hidden_states=z_1.to(self.pipe.transformer.dtype),
                    timestep=timestep / 1000.0,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0][:, :N, :]

                # We have source (original prompt) and target (edited prompt) branches
                v_src, v_trg = v_pred.chunk(2, dim=0)  # each [1, N, C]
                
                # Anneal alpha schedule for blending transported vs target velocity (u)
                alpha_k = alpha_start + (alpha_start - alpha_end) * k / (warm_start_steps - 1)

                h_avg = hook_mgr.get_averaged_hidden_states()  # [2, N, D]
                h_src_all, h_tgt_all = h_avg.chunk(2, dim=0)  # each [1, N, D]
                h_known = h_src_all[:, kn_idx, :]   # [1, K_n, D]
                h_unknown = h_tgt_all[:, unk_idx, :]  # [1, U, D]

                # Optimal transport step
                # Build OT transport cost matrix
                cost_matrix = build_transport_cost(
                    h_unknown=h_unknown,
                    h_known=h_known,
                    pos_unknown=pos_unk,
                    pos_known=pos_kn,
                    mask_2d=unknown_token_mask_2d,
                    lambda_pos=0.1,
                    lambda_bdry=5.0,
                    boundary_band_width=3, # Boundary (seam) width
                )

                # Solve OT problem and obtain plan (pi)
                plan = sinkhorn_transport(
                    cost_matrix,
                    tau=0.05,
                    num_iters=100,
                )

                # Transport known-region velocity to unknown tokens
                v_src_known = v_src[:, kn_idx, :]  # [1, K_n, C]

                # Weighted average to transport known-region source velocity to unknown tokens
                v_transported = torch.bmm(plan.to(v_src_known.dtype), v_src_known)

                # Unknown region velocity
                v_tgt_unk = v_trg[:, unk_idx, :]
                v_unknown = poisson_interp(
                    v_transported=v_transported,
                    v_tgt=v_tgt_unk,
                    v_src_full=v_src,
                    mask_2d=unknown_token_mask_2d,
                    alpha=alpha_k,
                    lambda_s=0.5,
                    token_hw=token_hw,
                    connectivity=8,
                )

                # Compose known and unknown region velocities together
                v_full = torch.zeros_like(v_src)
                v_full[:, unk_idx, :] = v_unknown.to(dtype)
                v_full_batch = v_full.expand(2, -1, -1)  # both branches get same update

                z_1 = z_1.float()
                
                # Update latent in masked (unknown) regions only
                z_1 = (1.0 - m) * z_1 + m * (z_1 + dt * v_full_batch.float())
                z_1 = z_1.to(dtype)
        
        finally:
            if hook_mgr is not None:
                hook_mgr.deregister()

        return z_1

    # Helper function to build reinjection mask for known region reinjections
    def _build_reinjection_mask(self, mask, token_hw, device, dtype, expand_unknown_pixels=4, dilate_tokens=0):
        reinjection_mask = reinject_mask_token_expanded_unknown(
            mask,
            token_hw,
            batch_size=1,
            device=device,
            dtype=dtype,
            expand_unknown_pixels=expand_unknown_pixels,
        )
        if dilate_tokens > 0:
            reinjection_mask = dilate_token_mask(
                reinjection_mask, token_hw, dilate_tokens
            )
        return reinjection_mask

    # Denoise image back to edited latent using Uni-Edit scheduler
    @torch.no_grad()
    def denoise_image(
        self,
        z_1, # Noisy latent (z_1 from report)
        unknown_mask,   # Unknown region token mask
        latent_image_ids, # Source and target latent IDs
        inversion_trajectory, # Intermediate latents for reinjection
        prompt_embeds, # Source and target prompt embeddings
        text_ids, # Source and target token IDs
        steps=50, # Denoising steps
        omega=7.0, # Guidance scale for Uni-Edit scheduler
        # Warm start steps
        warm_start_steps=10,
        known_mask=None, # Known region token mask for reinjection
        # Harmonisation mask and steps for geometric editing (last steps)
        harmonisation_mask=None,
        harmonisation_token_mask=None,
        late_steps=0
    ):
        # Token mask dimensions TODO: check
        tok_h, tok_w = unknown_mask.shape[-2], unknown_mask.shape[-1]

        # Retrieve image sequence length
        N = z_1.shape[1]

        # Set up Uni-Edit scheduler for denoising
        alpha = (steps - warm_start_steps) / steps if warm_start_steps > 0 else 1.0
        self.edit_scheduler.set_hyperparameters(alpha=alpha, omega=omega)
        self.edit_scheduler.set_mask_token_shape(tok_h, tok_w)
        self.edit_scheduler.set_external_guidance_mask(unknown_mask)

        # Set up timesteps
        sigmas_np = np.linspace(1.0, 1.0 / steps, steps)
        mu = _compute_empirical_mu(image_seq_len=N, num_steps=steps)
        self.edit_scheduler.set_timesteps(steps, device=z_1.device, sigmas=sigmas_np, mu=mu)
        timesteps = self.edit_scheduler.timesteps
        num_edit_steps = len(timesteps)
        self.edit_scheduler._step_index = 0
        self.edit_scheduler._begin_index = 0

        # Predicate to check whether to use harmonisation mask
        use_harmonisation_mask = harmonisation_mask is not None and harmonisation_token_mask is not None
        if use_harmonisation_mask:
            self.edit_scheduler.set_late_external_guidance_mask(
                harmonisation_token_mask,
                late_steps,
            )
        else:
            self.edit_scheduler.set_late_external_guidance_mask(None, 0)

        # Denoising loop
        z_t = z_1.clone()
        for j, t in tqdm(enumerate(timesteps), total=num_edit_steps, desc="Denoising"):
            step_mask = unknown_mask
            if use_harmonisation_mask and j >= num_edit_steps - late_steps:
                step_mask = harmonisation_token_mask

            # Index into inversion_trajectory: map edit step j -> inversion step
            # This is to factor in warm start steps from BCDM
            reinject_idx = steps - warm_start_steps - j
            reinject_idx = max(0, min(reinject_idx, len(inversion_trajectory) - 1))
            z_inv = inversion_trajectory[reinject_idx]
            z_inv_batch = z_inv.expand(2, -1, -1)

            # Reinject observed tokens from inversion trajectory
            z_t = step_mask * z_t + (1.0 - step_mask) * z_inv_batch

            # Transformer velocity prediction
            timestep = t.expand(z_t.shape[0]).to(z_t.dtype)
            noise_pred = self.pipe.transformer(
                hidden_states=z_t.to(self.pipe.transformer.dtype),
                timestep=timestep / 1000.0,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0][:, :N, :]

            # Scheduler step (UniEditEulerScheduler handles velocity fusion)
            z_t = self.edit_scheduler.step(noise_pred, t, z_t, return_dict=False)[0]

        # Final reinjection before VAE decoding
        if len(inversion_trajectory) > 0:
            z_inv_batch = inversion_trajectory[0].expand(2, -1, -1)
            step_mask = known_mask if use_harmonisation_mask else unknown_mask
            z_t = step_mask * z_t + (1.0 - step_mask) * z_inv_batch

        # Return denoised latent for decoding
        return z_t

    def run(
        self,
        image,
        prompt_src,
        prompt_tgt,
        steps=50,
        omega=7.0,
        warm_start_steps=10,
        unknown_mask=None, # Unknown region mask (numpy array)
        harmonisation_mask=None,
        late_steps=0,
        vae_color_fix=True
    ):
        # Return if unknown mask is not provided
        if unknown_mask is None:
            return image

        # Skip BCDM (for inpainting) if warm start steps is 0
        skip_bcdm = warm_start_steps <= 0
        
        # VAE encode
        image_latent, latent_ids = self.image_to_latent(image)
        # Invert image into noisy latent z_1
        z_1, inversion_trajectory = self.invert_image(image_latent, latent_ids, store_for_reinjection=True)
        # Build packed latent
        z_1_edit = torch.cat([z_1, z_1], dim=0)
        _, c, h, w = z_1_edit.shape
        # Number of image tokens
        N = h * w
        token_hw = (h, w)

        # Flatten latents for transformer
        z_1_packed = self.pipe._pack_latents(z_1_edit)

        # Build token-space mask
        token_mask = mask_to_token_space(unknown_mask, token_hw, batch_size=1, device=z_1_packed.device, dtype=z_1_packed.dtype)
        # Build 2D token-space mask
        token_mask_2d = mask_2d_from_token_mask(token_mask, token_hw)

        # Preprocess reinjection masks
        reinject_token_mask = self._build_reinjection_mask(
            unknown_mask,
            token_hw,
            z_1_packed.device, 
            z_1_packed.dtype,
            expand_unknown_pixels=4 if skip_bcdm else 8,
            dilate_tokens=1 if skip_bcdm else 0
        )
        if harmonisation_mask is not None:
            harmonisation_token_mask = self._build_reinjection_mask(
                harmonisation_mask,
                token_hw,
                z_1_packed.device,
                z_1_packed.dtype,
                expand_unknown_pixels=4,
                dilate_tokens=0
            )

        # Replace unknown region with Gaussian noise (naive initialisation)
        if not skip_bcdm:
            eps = torch.randn_like(z_1_packed[:1]) # Noise
            z_1_packed = (1. - token_mask) * z_1_packed + token_mask * torch.cat([eps, eps], dim=0)

        # Boundary-constrained distribution matching (BCDM)
        # Prepare prompt embeddings for editing
        prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=[prompt_src, prompt_tgt],
            device=z_1_packed.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        latent_image_ids = self.pipe._prepare_latent_ids(z_1_edit).to(z_1_packed.device)

        if not skip_bcdm:
            sigmas_np = np.linspace(1.0, 1.0 / steps, steps)
            mu = _compute_empirical_mu(image_seq_len=N, num_steps=steps)
            self.scheduler.set_timesteps(steps, device=z_1_packed.device, sigmas=sigmas_np, mu=mu)
            warm_start_sigmas = self.scheduler.sigmas[: warm_start_steps + 1].clone()

            # BCDM loop
            z_1_packed = self.bcdm_loop(
                z_1_packed,
                unknown_token_mask=token_mask,
                unknown_token_mask_2d=token_mask_2d,
                token_hw=token_hw,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                latent_image_ids=latent_image_ids,
                warm_start_steps=warm_start_steps,
                sigmas=warm_start_sigmas
            )
            z_1_packed = z_1_packed.to(z_1_edit.dtype)

        # Denoise latent
        z_0 = self.denoise_image(
            z_1=z_1_packed,
            unknown_mask=token_mask,
            latent_image_ids=latent_image_ids,
            inversion_trajectory=inversion_trajectory,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            steps=steps,
            omega=omega,
            warm_start_steps=warm_start_steps,
            known_mask=reinject_token_mask,
            harmonisation_mask=None,
            harmonisation_token_mask=None,
            late_steps=0
        )
        
        # Decode latent to image
        decoded_image = self.latent_to_image(
            z_0,
            latent_image_ids, 
            unknown_mask=unknown_mask,
            reference_image=image,
            vae_color_fix=vae_color_fix
        )

        return decoded_image

    @torch.no_grad()
    def run_geom_edit(
        self,
        src_image,
        tgt_image,
        spec: GeometrySpec,
        composition_prompt_callback: Optional[Callable[[Image.Image], Optional[str]]] = None, # Callback for composition prompt for demo
    ):
        # Pre: Source and target images are resized to same size (512x512 for our report)
        image_size = src_image.size

        # Parse inputs and edit types
        # If using multi-object composition, both source and target images and masks are required
        if spec.edit_type == EditType.COMPOSE_MULTI:
            if tgt_image is None:
                raise ValueError("Composition task requires a target image")
        else:
            tgt_image = src_image

        assert tgt_image is not None
        if spec.mask_tgt is None:
            raise ValueError("Geometry edit requires mask_tgt for composition / seam mask")

        # Prepare cases for source image inpainting.
        # If copy operation, simply return the source image
        if spec.edit_type == EditType.COPY:
            inpainted_image = src_image
        else:
            # If others, build inpainting mask and apply inpainting with our method
            inpaint_mask = build_inpainting_mask(spec)
            if bool(torch.any(inpaint_mask.detach().float() > 0.5)):
                inpaint_mask_np = geom_mask_to_np(inpaint_mask, image_size)
                # Run existing pipeline for inpainting
                inpainted_image = self.run(
                    src_image,
                    prompt_src=spec.prompt_inpaint,
                    prompt_tgt=spec.prompt_inpaint,
                    steps=50,
                    omega=7.0,
                    warm_start_steps=0, # Turn off BCDM since we only do inpainting
                    unknown_mask=inpaint_mask_np,
                    harmonisation_mask=None,
                    late_steps=0,
                    vae_color_fix=True
                )
            else:
                inpainted_image = src_image
        
        inpainted_image.save("outputs/inpainted_image.png")

        # After inpainting: apply composition if needed to obtain rough image I_c
        if spec.edit_type == EditType.COMPOSE_MULTI:
            # If use composition, we stack all masked transformed images on top of the base inpainted image
            composed_image = inpainted_image
            for T_i, paste_m in spec.compose_layers:
                coeffs = inverse_affine_coeffs_for_pil(T_i)
                warped = tgt_image.transform(
                    tgt_image.size,
                    Image.AFFINE,
                    coeffs,
                    resample=Image.BILINEAR,
                )
                paste_pil = mask_tensor_to_pil_l(paste_m, image_size)
                composite_mask = ImageOps.invert(paste_pil)
                composed_image = Image.composite(composed_image, warped, composite_mask).convert("RGB")
        else:
            # If transformation only, we only apply transform and compose with target image
            if spec.transform_matrix is not None:
                coeffs = inverse_affine_coeffs_for_pil(spec.transform_matrix)
                tgt_image = tgt_image.transform(
                    tgt_image.size,
                    Image.AFFINE,
                    coeffs,
                    resample=Image.BILINEAR,
                )
            mask_composite_pil = mask_tensor_to_pil_l(spec.mask_tgt, image_size)
            composite_mask = ImageOps.invert(mask_composite_pil)
            composed_image = Image.composite(inpainted_image, tgt_image, composite_mask).convert("RGB")

        composed_image.save("outputs/composed_image.png")

        # Refinement step: construct seam mask and apply BCDM again
        # We dilate the mask by a few pixels towards the outside to blend the object with the background
        band_k = 5
        blur_r = 2.0
        sigma_in = 1
        sigma_out = 12.0

        refine_mask_np = build_boundary_blur_mask(
            spec.mask_tgt,
            size_hw=image_size,
            # band_kernel_size=band_k,
            gaussian_radius=blur_r,
            # sigma_inside=sigma_in,
            # sigma_outside=sigma_out,
        )
        target_mask_np = geom_mask_to_np(spec.mask_tgt, image_size)
        # Expand to have the union of target mask and seam mask
        late_refine_mask_np = np.maximum(refine_mask_np, target_mask_np).astype(np.float32)
        late_edit_steps = 5

        # Build refinement prompt
        prompt_refine = spec.prompt_refine if spec.prompt_refine is not None else spec.prompt_inpaint
        if composition_prompt_callback is not None:
            callback_prompt = composition_prompt_callback(composed_image)
            if callback_prompt:
                prompt_refine = callback_prompt

        # Run our method again
        refined_image = self.run(
            composed_image,
            prompt_src=spec.prompt_inpaint,
            prompt_tgt=prompt_refine,
            steps=50,
            omega=7.0,
            warm_start_steps=0, # Turn off BCDM (optimal transport etc.)
            unknown_mask=refine_mask_np,
            harmonisation_mask=late_refine_mask_np,
            late_steps=late_edit_steps,
            vae_color_fix=True
        )
        return refined_image

# Hook manager for BCDM warm start vital layer feature extraction
class BCDMHookManager:
    def __init__(
        self,
        transformer,
        warm_layers,
        num_txt_tokens,
    ):
        self.transformer = transformer
        self.warm_layers = list(warm_layers)
        self.num_txt_tokens = num_txt_tokens
        self._handles: list = []
        self._captured: dict = {}

    # Register forward hooks for vital layers
    def register(self):
        # FLUX uses MMDiT architecture with double and single stream transformer blocks
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

    # Make forward hook for vital layers
    def _make_hook(self, key, block_type):
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
    
    # Get image-only hidden states for a block (capture image features only)
    def get_image_hidden_states(self, block_type, idx):
        return self._captured[(block_type, idx)]

    # Get aggregated average hidden states across all hooked layers
    def get_averaged_hidden_states(self):
        tensors = list(self._captured.values())
        if not tensors:
            raise RuntimeError("No hidden states captured")
        if len(tensors) == 1:
            return tensors[0]
        return torch.stack(tensors, dim=0).mean(dim=0)

    # Clear captured hidden states
    def clear(self):
        self._captured.clear()

    # Deregister forward hooks  
    def deregister(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._captured.clear()

# Helper function for color correction on decoded image to fix VAE decoding color drift.
# Per-channel affine (mean + std) color correction on decoded latent.
# Adapted from https://github.com/regiellis/ComfyUI-EasyColorCorrector/blob/main/src/nodes/vae_color_corrector.py
def vae_affine_color_fix_decoded(
        decoded: torch.Tensor,
        ref_tensor: torch.Tensor,
        mask_unknown_hw: torch.Tensor,
        min_pixels: int = 64,
        scale_lo: float = 0.5,
        scale_hi: float = 2.0,
    ) -> torch.Tensor:
        """Per-channel affine match ``decoded = scale * decoded + shift`` to ``ref`` on observed pixels.

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


# if __name__ == "__main__":
#     # Load image
#     image = Image.open("inputs/klein_snowman_scaled.png")
#     # Load mask
#     mask_image = Image.open("inputs/klein_25p_blur.png")
#     # Convert mask image to numpy array
#     mask_np = mask_image_to_numpy(mask_image)

#     # Invert image
#     pipe = BackbonePipeline()
#     decoded_image = pipe.run(image, "a snowman", "a snowman with pineapples and red house", unknown_mask=mask_np, warm_start_steps=0)
#     # Decode latent to image
#     decoded_image.save("outputs/decoded_image_inpainting.png")


if __name__ == "__main__":
    # Geometry edit smoke test (hardcoded from configs/geom_edit_pipeline.yaml / run_geom_edit.py).
    from typing import Any, Dict, Sequence

    _GEOM_CANVAS_HW = (512, 512)

    def _geom_load_mask_tensor(path: str) -> torch.Tensor:
        h, w = _GEOM_CANVAS_HW
        img = Image.open(path).convert("L").resize((w, h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _geom_paste_mask_centroid(paste_m: torch.Tensor) -> tuple[float, float]:
        m = paste_m.detach()
        while m.ndim > 3:
            m = m.squeeze(0)
        plane = m[0] if m.ndim == 3 else m
        H, W = plane.shape
        ys, xs = torch.where(plane > 0.5)
        if ys.numel() == 0:
            return (W - 1) * 0.5, (H - 1) * 0.5
        return float(xs.float().mean()), float(ys.float().mean())

    def _geom_affine_from_compose_entry(entry: Dict[str, Any], paste_m: torch.Tensor) -> torch.Tensor:
        if "matrix" in entry and entry["matrix"] is not None:
            t = torch.tensor(entry["matrix"], dtype=torch.float32)
            if t.shape != (3, 3):
                raise ValueError("'matrix' must be 3x3")
            return t
        s = float(entry.get("scale", 1.0))
        dx = float(entry.get("dx", 0.0))
        dy = float(entry.get("dy", 0.0))
        cx, cy = _geom_paste_mask_centroid(paste_m)
        tx = cx * (1.0 - s) + dx
        ty = cy * (1.0 - s) + dy
        return torch.tensor([[s, 0.0, tx], [0.0, s, ty], [0.0, 0.0, 1.0]], dtype=torch.float32)

    def _geom_spec_compose_multi() -> GeometrySpec:
        prompt_inpaint = (
            "Indoor table scene with soft, blurred background, with a log platter, plant vase and cup."
        )
        prompt_refine = (
            "A snowman with a yellow hat and scarf sitting on a log platter, against a soft, "
            "blurred table background. The scarf has white ends"
        )
        removal_mask = _geom_load_mask_tensor(
            "inputs/freefine/.freefine_checkout/Examples/Editing/2D/cake/source_mask.png"
        )
        compose_entry: Dict[str, Any] = {
            "paste_mask": "inputs/snowman_mask.png",
            "dy": -40,
            "scale": 0.75,
        }
        paste_m = _geom_load_mask_tensor(str(compose_entry["paste_mask"]))
        T = _geom_affine_from_compose_entry(compose_entry, paste_m)
        return GeometrySpec.for_compose_multi(
            [removal_mask],
            [(T, paste_m)],
            prompt_inpaint,
            prompt_refine,
            mask_user=None,
        )

    src_path = "inputs/freefine/.freefine_checkout/Examples/Editing/2D/cake/source.png"
    tgt_path = "inputs/snowman.jpg"
    outdir = "outputs/geom_edit_pipeline"
    os.makedirs(outdir, exist_ok=True)

    spec = _geom_spec_compose_multi()
    src_image = Image.open(src_path).convert("RGB").resize((512, 512))
    tgt_image = Image.open(tgt_path).convert("RGB").resize((512, 512))

    pipe = BackbonePipeline()
    result = pipe.run_geom_edit(src_image, tgt_image, spec)
    result.save(os.path.join(outdir, "refined.png"))
    print(f"[pipeline] Geometry edit done. Saved to {outdir}/refined.png")

