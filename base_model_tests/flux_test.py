import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
# from dfloat11 import DFloat11Model
from diffusers.utils import load_image

image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

# # pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
# pipe = FluxFillPipeline.from_pretrained("YarvixPA/FLUX.1-Fill-dev-GGUF", model_file= torch_dtype=torch.uint8)
# # pipe.enable_sequential_cpu_offload()

# # DFloat11Model.from_pretrained('DFloat11/FLUX.1-Fill-dev-DF11', device='cuda:1', bfloat16_model=pipe.transformer)

transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-gguf/blob/main/flux1-fill-dev-Q4_0.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
device = torch.device("cuda:1")
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to(device)
# pipe.enable_model_cpu_offload()

image = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=576,
    width=576,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"flux-fill-dev.png")