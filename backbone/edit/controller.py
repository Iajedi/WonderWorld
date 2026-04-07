"""BCOTHVEPipeline -- orchestrator for BCOT-HVE inpainting / outpainting.

Wraps the existing Flux2UniEditFlowPipeline (or FluxUniEditFlowPipeline)
and inserts a K-step warm-start phase before the standard UniEdit-Flow
editing loop.  No existing scheduler or pipeline code is modified.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from edit.warm_start import warm_start_loop
from utils.mask_ops import (
    infer_token_hw,
    mask_to_token_space,
    resize_mask_to_latent,
    mask_2d_from_token_mask,
    extract_boundary_band,
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
    """

    def __init__(self, offload: bool = False, model: str = "klein"):
        if model == "klein":
            from flux_pipeline import Flux2UniEditFlowPipeline
            self.wrapper = Flux2UniEditFlowPipeline(offload=offload)
        else:
            from flux_pipeline import FluxUniEditFlowPipeline
            self.wrapper = FluxUniEditFlowPipeline(offload=offload)
        self.model_type = model

    @torch.no_grad()
    def run(
        self,
        image: Union[str, Image.Image],
        mask: Union[str, np.ndarray, torch.Tensor],
        prompt_src: str,
        prompt_tgt: str,
        config: Dict[str, Any],
        output_dir: str = "outputs/bcot_hve",
    ) -> Image.Image:
        """Execute the full four-phase BCOT-HVE pipeline.

        Returns the final decoded PIL image.
        """
        os.makedirs(output_dir, exist_ok=True)

        T = int(config.get("T", 50))
        K = int(config.get("K", 10))
        warm_method = config.get("warm_method", "ot_harmonic")
        omega = float(config.get("omega", 5.0))
        alpha_edit = float(config.get("alpha_edit", 0.8))
        skip_alpha = (T - K) / T if K > 0 else alpha_edit
        debug = config.get("debug", False)

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

        # Black out unknown region to prevent info leakage
        mask_pil = Image.fromarray((mask_np.squeeze() * 255).astype(np.uint8), mode="L")
        bg = Image.new("RGB", image.size, (0, 0, 0))
        image = Image.composite(bg, image.convert("RGB"), mask_pil.resize(image.size))

        # VAE encode
        image_latent, latent_ids = self.wrapper.image2latent(image)
        # image_latent is [1, C, H_tok, W_tok] (4-D patchified)

        # Invert for alpha * T steps
        from uniedit_flow_schedulers.UniInvEulerScheduler import UniInvEulerScheduler
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

        # Replace unknown region with Gaussian noise
        eps = torch.randn_like(z_packed[:1])  # [1, N, C]
        m = mask_token  # [1, N, 1]
        z_packed = (1.0 - m) * z_packed + m * torch.cat([eps, eps], dim=0)

        if debug:
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
            debug_warmdir = os.path.join(output_dir, "warm_debug") if debug else None

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

        if debug:
            torch.save(z_packed.cpu(), os.path.join(output_dir, "z_after_warmstart.pt"))

        # ------------------------------------------------------------------
        # Phase C: standard UniEdit-Flow for the remaining T-K steps
        # ------------------------------------------------------------------
        # Unpack from [2, N, C] back to [2, C, H_tok, W_tok]
        z_4d = z_packed.permute(0, 2, 1).reshape(2, C_lat, tok_h, tok_w)

        self.wrapper.edit_scheduler.set_hyperparameters(alpha=skip_alpha, omega=omega)

        self.wrapper.edit_scheduler.set_debug_options(
            save_masks=debug,
            print_mask_stats=False,
            mask_save_every=5,
            mask_dir=os.path.join(output_dir, "edit_masks"),
        )
        self.wrapper.edit_scheduler.set_mask_token_shape(tok_h, tok_w)

        from flux_pipeline import _manual_mask_to_token_space
        external_mask = _manual_mask_to_token_space(
            manual_mask=mask_np,
            latent_hw=(tok_h, tok_w),
            batch_size=1,
            device=z_4d.device,
            dtype=z_4d.dtype,
        )
        self.wrapper.edit_scheduler.set_external_guidance_mask(external_mask)

        pipe.scheduler = self.wrapper.edit_scheduler
        result_image = pipe(
            prompt=[prompt_src, prompt_tgt],
            num_inference_steps=T,
            guidance_scale=1.0,
            latents=z_4d.to(dtype),
            max_sequence_length=512,
            height=512,
            width=512,
        ).images[0]

        # ------------------------------------------------------------------
        # Phase D: save outputs
        # ------------------------------------------------------------------
        result_image.save(os.path.join(output_dir, "result.png"))

        if debug:
            self._save_debug_outputs(image, result_image, mask_np, token_hw, output_dir)

        return result_image

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
        output_dir: str,
    ) -> None:
        original.save(os.path.join(output_dir, "original_masked.png"))
        m_vis = (mask_np.squeeze() * 255).astype(np.uint8)
        Image.fromarray(m_vis, mode="L").save(os.path.join(output_dir, "mask.png"))

        metrics = self.compute_metrics(original, result, mask_np)
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")
        print(f"[BCOT-HVE] Metrics: {metrics}")
