import torch
from PIL import Image
from diffusers import QwenImageEditInpaintPipeline
from diffusers.utils import load_image

pipe = QwenImageEditInpaintPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
pipe.to("cuda")
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
source = load_image(img_url)

mask = load_image(mask_url)
image = pipe(prompt=prompt, negative_prompt=" ", image=source, mask_image=mask, strength=1.0, num_inference_steps=50).images[0]

image.save("qwenimage_inpainting.png")
