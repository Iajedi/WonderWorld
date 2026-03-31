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

class FluxStableFlowPipeline:
    def __init__(self, offload=False):
        self.pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype)
        if offload:
            self.pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
        else:
            print(torch.cuda.is_available())
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

if __name__=="__main__":
    pipe = FluxStableFlowPipeline(offload=False)
    pipe.generate(prompt="Tokyo Tower looming over a night cityscape in Japan", seed=42)


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
