import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline, FluxPipeline
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
import os
import numpy as np
from PIL import Image

device = "cuda"
dtype = torch.bfloat16

MM_VITAL_BLOCKS = list([1, 2])
# MM_VITAL_BLOCKS = []
NUM_MM_BLOCKS = 5
SINGLE_VITAL_BLOCKS = list(np.array([9, 10, 15, 24]) - NUM_MM_BLOCKS)

def mask_image(image):
    mask_image = Image.open('inputs/klein_25p_blur.png')
    mask = mask_image.convert("L")
    background = Image.new("RGB", image.size, (0, 0, 0))
    image = image.convert("RGB")
    image = Image.composite(background, image, mask)
    return image

def image_to_mask(image, h=512, w=512):
    image = image.convert("L")
    image = image.resize((h, w))
    image = np.array(image)
    image = image / 255.0
    image = image.astype(np.float32)
    # Reshape to [1, 1, H, W]
    mask = image.reshape(1, 1, h, w)
    return mask

# SINGLE_VITAL_BLOCKS = list(range(20))
def _manual_mask_to_token_space(manual_mask, latent_hw, batch_size, device, dtype):
    if manual_mask is None:
        return None

    if isinstance(manual_mask, np.ndarray):
        mask = torch.from_numpy(manual_mask)
    elif isinstance(manual_mask, torch.Tensor):
        mask = manual_mask
    else:
        raise TypeError(f"manual_mask must be a numpy array or torch tensor, got {type(manual_mask)}")

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask.unsqueeze(0)
        elif mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("manual_mask with 3 dims must be [1, H, W] or [H, W, 1]")
    elif mask.ndim == 4:
        if mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError("manual_mask with 4 dims must be [1, 1, H, W]")
    else:
        raise ValueError(f"manual_mask must have shape [H,W], [1,H,W], [H,W,1], or [1,1,H,W], got {mask.shape}")

    mask = mask.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    latent_h, latent_w = int(latent_hw[0]), int(latent_hw[1])
    mask = F.interpolate(mask, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    mask = mask.reshape(1, latent_h * latent_w, 1)
    if batch_size > 1:
        mask = mask.expand(batch_size, -1, -1)
    return mask.to(device=device, dtype=dtype)


class Flux2StableFlowPipeline:
    def __init__(self, offload=False, device: str = "cuda"):
        self.device = str(device)
        self.pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype)
        if offload:
            self.pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
        else:
            self.pipe.to(self.device)

    @torch.no_grad()
    def generate(self, prompt, seed):
        image = self.pipe(
            prompt=prompt,
            height=512,
            width=512,
            guidance_scale=5.0,
            num_inference_steps=50,
            generator=torch.Generator(device=self.device).manual_seed(seed)
        ).images[0]

        return image

    @torch.no_grad()
    def image2latent(self, image, latent_nudging_scalar = 1.0):
        image = self.pipe.image_processor.preprocess(image).type(self.pipe.vae.dtype).to(self.device)
        latents = self.pipe._encode_vae_image(image, generator=None)
        latents = latents * latent_nudging_scalar

        return latents
    
    @torch.no_grad()
    def invert_and_save(self, image, prompts):
        inversion_prompt = prompts[0:1]
        # Invert
        inv_image, inverted_latent_list, latent_ids = self.pipe(
            prompt=inversion_prompt,
            height=512,
            width=512,
            guidance_scale=1.0,
            num_inference_steps=50,
            max_sequence_length=512,
            latents=self.image2latent(image),
            invert_image=True
        )

        # Unpack latents using ids, tile
        inverted_latent = self.pipe._unpack_latents_with_ids(inverted_latent_list[-1], latent_ids)

        # Debug: image
        inv_image_decode = self.pipe.vae.decode(inv_image, return_dict=False)[0]
        inv_image_decoded = self.pipe.image_processor.postprocess(inv_image_decode, output_type="pil")[0]
        inv_image_decoded.save("inv_klein2.png")

        # Prompt only: set to 0.4, ignore MM blocks
        t = 1
        edit_latent = t * inverted_latent + (1 - t) * torch.randn_like(inverted_latent)
        inverted_latents = torch.cat([inverted_latent, edit_latent])

        # Edit
        images = self.pipe(
            prompt=prompts,
            height=512,
            width=512,
            guidance_scale=5.0,
            num_inference_steps=50,
            max_sequence_length=512,
            latents=inverted_latents,
            inverted_latent_list=inverted_latent_list,
            mm_copy_blocks=MM_VITAL_BLOCKS,
            single_copy_blocks=SINGLE_VITAL_BLOCKS,
        ).images
        images = [np.array(img) for img in images]
        res = Image.fromarray(np.hstack((images)))
        res.save("edited.png")

class Flux2UniEditFlowPipeline:
    def __init__(self, offload=False, device: str = "cuda"):
        self.device = str(device)
        model_path = "black-forest-labs/FLUX.2-klein-base-4B"
        self.pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype)
        self.invert_scheduler = UniInvEulerScheduler.from_config(self.pipe.scheduler.config)
        self.edit_scheduler = UniEditEulerScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        if offload:
            self.pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
        else:
            self.pipe.to(self.device)

    @torch.no_grad()
    def image2latent(self, image, latent_nudging_scalar=1.0):
        '''image: PIL.Image'''
        image = self.pipe.image_processor.preprocess(image).type(dtype).to(self.device)
        latents = self.pipe._encode_vae_image(image, generator=None)
        latent_ids = self.pipe._prepare_latent_ids(latents)
        return latents * latent_nudging_scalar, latent_ids

    def _patchify_and_bn_normalize(self, latents):
        """Re-patchify output latents and re-apply BN normalization.

        The pipeline's output_type='latent' path applies BN denormalization and
        unpatchification. This reverses both so latents can be fed back into the
        pipeline for editing/reconstruction.
        """
        latents = self.pipe._patchify_latents(latents)
        bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = (latents - bn_mean) / bn_std
        return latents

    @torch.no_grad()
    def invert_and_save(
        self,
        image,
        prompts,
        alpha=0.5,
        omega=5.0,
        steps=50,
        manual_mask=None,
        debug_save_masks=False,
        debug_print_mask_stats=False,
        mask_save_every=5,
        mask_output_dir="outputs/masks",
    ):
        self.invert_scheduler.set_hyperparameters(alpha=alpha)
        self.edit_scheduler.set_hyperparameters(alpha=alpha, omega=omega)

        # Encode
        image_latent, latent_ids = self.image2latent(image)
        # Invert
        self.pipe.scheduler = self.invert_scheduler
        invert_noise_latent = self.pipe(
            prompt="",
            num_inference_steps=steps,
            guidance_scale=1.0,
            latents=image_latent.to(dtype),
            output_type='latent',
            height=512,
            width=512
        ).images

        invert_noise_latent = self._patchify_and_bn_normalize(invert_noise_latent)
        edit_init_latent = torch.cat([invert_noise_latent, invert_noise_latent])
        token_h, token_w = edit_init_latent.shape[-2], edit_init_latent.shape[-1]

        self.edit_scheduler.set_debug_options(
            save_masks=debug_save_masks,
            print_mask_stats=debug_print_mask_stats,
            mask_save_every=mask_save_every,
            mask_dir=mask_output_dir,
        )
        self.edit_scheduler.set_mask_token_shape(token_h, token_w)

        external_mask = _manual_mask_to_token_space(
            manual_mask=manual_mask,
            latent_hw=(token_h, token_w),
            batch_size=edit_init_latent.shape[0] // 2,
            device=edit_init_latent.device,
            dtype=edit_init_latent.dtype,
        )
        self.edit_scheduler.set_external_guidance_mask(external_mask)

        # Edit
        self.pipe.scheduler = self.edit_scheduler
        recon_image = self.pipe(
            prompt=prompts,
            num_inference_steps=steps,
            guidance_scale=1.0,
            latents=edit_init_latent.to(dtype),
            max_sequence_length=512,
            height=512,
            width=512
        ).images[0]

        recon_image.save('edited.png')

    @torch.no_grad()
    def invert_and_recon(self, image, prompt, alpha=1.0, steps=50):
        self.invert_scheduler.set_hyperparameters(alpha=alpha)

        # Encode
        image_latent, latent_ids = self.image2latent(image)

        # Invert
        self.pipe.scheduler = self.invert_scheduler
        invert_noise_latent = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=1.0,
            latents=image_latent.to(dtype),
            output_type='latent',
            height=512,
            width=512
        ).images

        # Debug: image
        inv_image_decode = self.pipe.vae.decode(invert_noise_latent, return_dict=False)[0]
        inv_image_decoded = self.pipe.image_processor.postprocess(inv_image_decode.detach(), output_type="pil")[0]
        inv_image_decoded.save("inv_klein2.png")

        invert_noise_latent = self._patchify_and_bn_normalize(invert_noise_latent)

        # Recon
        self.pipe.scheduler = self.scheduler
        recon_image = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=1.0,
            latents=invert_noise_latent,
            max_sequence_length=512,
            height=512,
            width=512
        ).images[0]

        recon_image.save('recon.png')

class FluxUniEditFlowPipeline:
    def __init__(self, offload=False, device: str = "cuda"):
        self.device = str(device)
        model_path = "black-forest-labs/FLUX.1-dev"
        self.pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
        self.invert_scheduler = UniInvEulerScheduler.from_config(self.pipe.scheduler.config)
        self.edit_scheduler = UniEditEulerScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        if offload:
            self.pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
        else:
            self.pipe.to(self.device)

    @torch.no_grad()
    def image2latent(self, image, latent_nudging_scalar=1.0):
        '''image: PIL.Image'''
        image = self.pipe.image_processor.preprocess(image).type(dtype).to(self.device)
        latents = self.pipe._encode_vae_image(image, generator=None)
        latent_ids = self.pipe._prepare_latent_ids(latents)
        return latents * latent_nudging_scalar, latent_ids

    @torch.no_grad()
    def invert_and_save(
        self,
        image,
        prompt,
        alpha=0.6,
        omega=5.0,
        steps=20,
        manual_mask=None,
        debug_save_masks=False,
        debug_print_mask_stats=False,
        mask_save_every=5,
        mask_output_dir="outputs/masks",
    ):
        self.invert_scheduler.set_hyperparameters(alpha=alpha)
        self.edit_scheduler.set_hyperparameters(alpha=alpha, omega=omega)
        self.edit_scheduler.set_debug_options(
            save_masks=debug_save_masks,
            print_mask_stats=debug_print_mask_stats,
            mask_save_every=mask_save_every,
            mask_dir=mask_output_dir,
        )
        # Encode
        image_latent, latent_ids = self.image2latent(image)
        # Invert
        self.pipe.scheduler = self.invert_scheduler
        invert_noise_latent = self.pipe(
            prompt="",
            num_inference_steps=steps,
            guidance_scale=1.0,
            latents=image_latent.to(dtype),
            output_type='latent',
            height=512,
            width=512
        ).images

        invert_noise_latent = self.pipe._patchify_latents(invert_noise_latent)
        edit_init_latent = torch.cat([invert_noise_latent, invert_noise_latent])
        token_h, token_w = edit_init_latent.shape[-2], edit_init_latent.shape[-1]

        self.edit_scheduler.set_debug_options(
            save_masks=debug_save_masks,
            print_mask_stats=debug_print_mask_stats,
            mask_save_every=mask_save_every,
            mask_dir=mask_output_dir,
        )
        self.edit_scheduler.set_mask_token_shape(token_h, token_w)

        external_mask = _manual_mask_to_token_space(
            manual_mask=manual_mask,
            latent_hw=(token_h, token_w),
            batch_size=edit_init_latent.shape[0] // 2,
            device=edit_init_latent.device,
            dtype=edit_init_latent.dtype,
        )
        self.edit_scheduler.set_external_guidance_mask(external_mask)

        # Edit
        self.pipe.scheduler = self.edit_scheduler
        recon_image = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=1.0,
            latents=edit_init_latent.to(dtype),
            max_sequence_length=512,
            height=512,
            width=512
        ).images[0]

        recon_image.save('edited_flux1.png')

    @torch.no_grad()
    def invert_and_recon(self, image, prompts, alpha=1.0, steps=50):
        self.invert_scheduler.set_hyperparameters(alpha=alpha)

        # Encode
        image_latent, latent_ids = self.image2latent(image)

        # Invert
        self.pipe.scheduler = self.invert_scheduler
        invert_noise_latent = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=4.0,
            latents=image_latent.to(dtype),
            output_type='latent',
            height=512,
            width=512
        ).images

        # Debug: image
        inv_image_decode = self.pipe.vae.decode(invert_noise_latent, return_dict=False)[0]
        inv_image_decoded = self.pipe.image_processor.postprocess(inv_image_decode, output_type="pil")
        inv_image_decoded.save("inv_klein.png")

        invert_noise_latent = self.pipe._patchify_latents(invert_noise_latent)

        # Recon
        self.pipe.scheduler = self.scheduler
        recon_image = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=4.0,
            latents=invert_noise_latent,
            max_sequence_length=512,
            height=512,
            width=512
        ).images[0]

        recon_image.save('recon.png')
    

if __name__=="__main__":
    pipe = Flux2UniEditFlowPipeline(offload=False)
    mask = image_to_mask(Image.open('inputs/klein_25p_blur.png'))

    prompt1 = "A cheerful snowman stands on a lakeside promenade in a Mediterranean-style coastal village, holding a sign that reads \"holl world.\" Seven pineapples are lined up along the stone ledge in front of the snowman, with turquoise water and mountainous terrain in the background."
    prompt2 = "A cheerful snowman stands on a lakeside promenade in a Mediterranean-style coastal village, holding a sign that reads \"holl world.\" Seven pineapples are lined up along the stone ledge in front of the snowman, with turquoise water and mountainous terrain in the background."
    # prompt2 = "A lakeside promenade in a Mediterranean-style coastal village. Seven pineapples are lined up along the stone ledge in front, with turquoise water and mountainous terrain in the background."
    image = Image.open('inputs/klein_snowman_scaled.png')
    image = mask_image(image)

    image.save('temp.png')
    pipe.invert_and_save(
        image, prompts=[prompt1, prompt2], alpha=0.8, omega=5.0, steps=50,
        debug_save_masks=True,
        debug_print_mask_stats=False,
        manual_mask=mask,
        mask_save_every=5,
        mask_output_dir="outputs/masks_outpaint",
    )
    # pipe.invert_and_save(image, prompts=["Tokyo Tower at night", "Burj Khalifa at night"])
    # pipe.invert_and_recon(image, prompt="Statue of liberty", steps=50)

    # pipe = FluxStableFlowPipeline
    # image = Image.open('flux-klein.png')

# @torch.no_grad()
# def image2latent(self, image, latent_nudging_scalar = 1.15):
#     image = self.pipe.image_processor.preprocess(image).type(self.pipe.vae.dtype).to("cuda")
#     latents = self.pipe.vae.encode(image)["latent_dist"].mean
#     latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
#     latents = latents * latent_nudging_scalar
#     latents = self.pipe._pack_latents(
#         latents=latents,
#         batch_size=1,
#         num_channels_latents=16,
#         height=128,
#         width=128
#     )

#     return latents

# prompt = "Tokyo Tower looming over a night cityscape in Japan"
# image = pipe(
#     prompt=prompt,
#     height=512,
#     width=512,
#     guidance_scale=5.0,
#     num_inference_steps=50,
#     generator=torch.Generator(device=device).manual_seed(42)
# ).images[0]
# image.save("flux-klein.png")
