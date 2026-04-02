import torch
from diffusers import Flux2KleinPipeline, FluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from uniedit_flow_schedulers.UniInvEulerScheduler import UniInvEulerScheduler
from uniedit_flow_schedulers.UniEditEulerScheduler import UniEditEulerScheduler
import os
import numpy as np
from PIL import Image

device = "cuda"
dtype = torch.bfloat16

MM_VITAL_BLOCKS = []
NUM_MM_BLOCKS = 5
# SINGLE_VITAL_BLOCKS = list(np.array([9, 10, 15, 24]) - NUM_MM_BLOCKS)

SINGLE_VITAL_BLOCKS = []
class Flux2StableFlowPipeline:
    def __init__(self, offload=False):
        self.pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype)
        if offload:
            self.pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
        else:
            # print(torch.cuda.is_available())
            self.pipe.to(device)

    @torch.no_grad
    def generate(self, prompt, seed):
        image = self.pipe(
            prompt=prompt,
            height=512,
            width=512,
            guidance_scale=5.0,
            num_inference_steps=50,
            generator=torch.Generator(device=device).manual_seed(seed)
        ).images[0]

        return image

    @torch.no_grad()
    def image2latent(self, image, latent_nudging_scalar = 1.0):
        image = self.pipe.image_processor.preprocess(image).type(self.pipe.vae.dtype).to(device)
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
    def __init__(self, offload=False):
        model_path = "black-forest-labs/FLUX.2-klein-base-4B"
        self.pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype)
        self.invert_scheduler = UniInvEulerScheduler()
        self.edit_scheduler = UniEditEulerScheduler()
        self.scheduler = FlowMatchEulerDiscreteScheduler()

        if offload:
            self.pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
        else:
            # print(torch.cuda.is_available())
            self.pipe.to(device)

    @torch.no_grad()
    def image2latent(self, image, latent_nudging_scalar=1.0):
        '''image: PIL.Image'''
        image = self.pipe.image_processor.preprocess(image).type(dtype).to(device)
        latents = self.pipe._encode_vae_image(image, generator=None)
        latent_ids = self.pipe._prepare_latent_ids(latents)
        return latents * latent_nudging_scalar, latent_ids

    @torch.no_grad()
    def invert_and_save(self, image, prompt, alpha=1.0, omega=5.0, steps=20):
        self.invert_scheduler.set_hyperparameters(alpha=alpha)
        self.edit_scheduler.set_hyperparameters(alpha=alpha, omega=omega)

        # Encode
        image_latent, latent_ids = self.image2latent(image)
        # Invert
        pipe.scheduler = self.invert_scheduler
        invert_noise_latent = self.pipe(
            prompt="",
            num_inference_steps=steps,
            guidance_scale=1.0,
            latents=image_latent.to(dtype),
            output_type='latent',
            height=512,
            width=512
        ).images

        # TODO: simplify patchify and unpatchify logic
        invert_noise_latent = self.pipe._patchify_latents(invert_noise_latent)
        edit_init_latent = torch.cat([invert_noise_latent, invert_noise_latent])

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

        recon_image.save('edited.png')

    @torch.no_grad
    def invert_and_recon(self, image, prompt, alpha=1.0, steps=50):
        self.invert_scheduler.set_hyperparameters(alpha=alpha)

        # Encode
        image_latent, latent_ids = self.image2latent(image)

        # Invert
        pipe.scheduler = self.invert_scheduler
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
        inv_image_decoded = self.pipe.image_processor.postprocess(inv_image_decode.detach(), output_type="pil")[0]
        inv_image_decoded.save("inv_klein2.png")

        # TODO: simplify patchify and unpatchify logic
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

class FluxUniEditFlowPipeline:
    def __init__(self, offload=False):
        model_path = "black-forest-labs/FLUX.1-dev"
        self.pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
        self.invert_scheduler = UniInvEulerScheduler()
        self.edit_scheduler = UniEditEulerScheduler()
        self.scheduler = FlowMatchEulerDiscreteScheduler()

        if offload:
            self.pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
        else:
            # print(torch.cuda.is_available())
            self.pipe.to(device)

    @torch.no_grad()
    def image2latent(self, image, latent_nudging_scalar=1.0):
        '''image: PIL.Image'''
        image = self.pipe.image_processor.preprocess(image).type(dtype).to(device)
        latents = self.pipe._encode_vae_image(image, generator=None)
        latent_ids = self.pipe._prepare_latent_ids(latents)
        return latents * latent_nudging_scalar, latent_ids

    @torch.no_grad()
    def invert_and_save(self, image, prompt, alpha=0.6, omega=5.0, steps=20):
        self.invert_scheduler.set_hyperparameters(alpha=alpha)
        self.edit_scheduler.set_hyperparameters(alpha=alpha, omega=omega)

        # Encode
        image_latent, latent_ids = self.image2latent(image)
        # Invert
        pipe.scheduler = self.invert_scheduler
        invert_noise_latent = self.pipe(
            prompt="",
            num_inference_steps=steps,
            guidance_scale=1.0,
            latents=image_latent.to(dtype),
            output_type='latent',
            height=512,
            width=512
        ).images

        # TODO: simplify patchify and unpatchify logic
        invert_noise_latent = self.pipe._patchify_latents(invert_noise_latent)
        edit_init_latent = torch.cat([invert_noise_latent, invert_noise_latent])

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

    @torch.no_grad
    def invert_and_recon(self, image, prompt, alpha=1.0, steps=50):
        self.invert_scheduler.set_hyperparameters(alpha=alpha)

        # Encode
        image_latent, latent_ids = self.image2latent(image)

        # Invert
        pipe.scheduler = self.invert_scheduler
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
        inv_image_decoded = self.pipe.image_processor.postprocess(inv_image_decode, output_type="pil")[0]
        inv_image_decoded.save("inv_klein.png")

        # TODO: simplify patchify and unpatchify logic
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
    # prompts = ["", ""]
    # prompt = "Tokyo Tower in a night cityscape"
    # image = pipe.generate(prompts[1], 42)
    
    image = Image.open('imperial_scaled.jpg')
    # pipe.invert_and_save(image, prompts=prompts)
    pipe.invert_and_recon(image, prompt="")

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
