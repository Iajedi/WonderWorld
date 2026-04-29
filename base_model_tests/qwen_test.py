# import torch
# from diffusers.utils import load_image

# # pip install git+https://github.com/huggingface/diffusers
# from diffusers import QwenImageControlNetModel, QwenImageControlNetInpaintPipeline

# base_model = "Qwen/Qwen-Image"
# controlnet_model = "InstantX/Qwen-Image-ControlNet-Inpainting"

# controlnet = QwenImageControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
# pipe = QwenImageControlNetInpaintPipeline.from_pretrained(
#     base_model, controlnet=controlnet, torch_dtype=torch.bfloat16, device_map="cuda"
# )
# pipe.enable_sequential_cpu_offload()

# image = load_image("https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/images/image1.png")
# mask_image = load_image("https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/masks/mask1.png")
# prompt = "一辆绿色的出租车行驶在路上"

# image = pipe(
#     prompt=prompt,
#     negative_prompt=" ",
#     control_image=image,
#     control_mask=mask_image,
#     controlnet_conditioning_scale=controlnet_conditioning_scale,
#     width=control_image.size[0],
#     height=control_image.size[1],
#     num_inference_steps=30,
#     true_cfg_scale=4.0,
#     generator=torch.Generator(device="cuda").manual_seed(42),
# ).images[0]
# image.save(f"qwenimage_cn_inpaint_result.png")

from diffusers import DiffusionPipeline, QwenImageTransformer2DModel
import torch
from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using Qwen-Image model')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offloading')
    parser.add_argument('--cpu_offload_blocks', type=int, default=None, help='Number of transformer blocks to offload to CPU')
    parser.add_argument('--no_pin_memory', action='store_true', help='Disable memory pinning')
    parser.add_argument('--prompt', type=str, default='A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".',
                        help='Text prompt for image generation')
    parser.add_argument('--negative_prompt', type=str, default=' ',
                        help='Negative prompt for image generation')
    parser.add_argument('--aspect_ratio', type=str, default='16:9', choices=['1:1', '16:9', '9:16', '4:3', '3:4'],
                        help='Aspect ratio of generated image')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--true_cfg_scale', type=float, default=4.0,
                        help='Classifier free guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for generation')
    parser.add_argument('--output', type=str, default='example.png',
                        help='Output image path')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'],
                        help='Language for positive magic prompt')
    return parser.parse_args()

args = parse_args()

model_name = "Qwen/Qwen-Image"

with no_init_weights():
    transformer = QwenImageTransformer2DModel.from_config(
        QwenImageTransformer2DModel.load_config(
            model_name, subfolder="transformer",
        ),
    ).to(torch.bfloat16)

DFloat11Model.from_pretrained(
    "DFloat11/Qwen-Image-DF11",
    device="cpu",
    cpu_offload=args.cpu_offload,
    cpu_offload_blocks=args.cpu_offload_blocks,
    pin_memory=not args.no_pin_memory,
    bfloat16_model=transformer,
)

pipe = DiffusionPipeline.from_pretrained(
    model_name,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt,
    "zh": "超清，4K，电影级构图" # for chinese prompt,
}

# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
}

width, height = aspect_ratios[args.aspect_ratio]

image = pipe(
    prompt=args.prompt + positive_magic[args.language],
    negative_prompt=args.negative_prompt,
    width=width,
    height=height,
    num_inference_steps=args.num_inference_steps,
    true_cfg_scale=args.true_cfg_scale,
    generator=torch.Generator(device="cuda").manual_seed(args.seed)
).images[0]

image.save(args.output)

max_memory = torch.cuda.max_memory_allocated()
print(f"Max memory: {max_memory / (1000 ** 3):.2f} GB")