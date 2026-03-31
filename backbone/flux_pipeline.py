import torch
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

# Do not use distilled model
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

# No of blocks: 25
# 5 double, 20 single per pass
print(len(pipe.transformer.transformer_blocks))
print(len(pipe.transformer.single_transformer_blocks))

# For k = 64
# We generate k seeds
# For each of the k prompts by GPT, fixed seed i
# Pass through model with all layers, fixed seed i, encode DinoV2
# Pass through model with one layer deactivated i, encode DinoV2
# Total: 64 * (25 + 1) iterations

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
