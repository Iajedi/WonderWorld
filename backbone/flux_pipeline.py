import torch
from diffusers import Flux2KleinPipeline
import os
import numpy as np
from PIL import Image

device = "cuda"
dtype = torch.bfloat16

MM_VITAL_BLOCKS = [0, 4]
NUM_MM_BLOCKS = 5
SINGLE_VITAL_BLOCKS = list(np.array([9, 10, 15, 24]) - NUM_MM_BLOCKS)
# SINGLE_VITAL_BLOCKS = []
class FluxStableFlowPipeline:
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

    @torch.no_grad()
    def image2latent(self, image, latent_nudging_scalar = 0.45):
        image = self.pipe.image_processor.preprocess(image).type(self.pipe.vae.dtype).to(device)
        latents = self.pipe._encode_vae_image(image, generator=None)
        latents = latents * latent_nudging_scalar

        return latents
    
    @torch.no_grad()
    def invert_and_save(self, image, prompts):
        inversion_prompt = prompts[0:1]
        # Invert
        inverted_latent_list, latent_ids = self.pipe(
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
        t = 1
        # Prompt only: set to 0.4, ignore MM blocks
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




if __name__=="__main__":
    pipe = FluxStableFlowPipeline(offload=False)
    prompts = ["Tokyo Tower in a night cityscape", "Statue of Liberty in a night cityscape"]
    image = Image.open("flux-klein.png")
    pipe.invert_and_save(image, prompts)

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
