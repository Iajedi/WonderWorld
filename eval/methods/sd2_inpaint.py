"""Stable Diffusion 2 inpainting adapter (Hugging Face)."""

from __future__ import annotations

import torch
from PIL import Image

from eval.methods._hf_common import make_generator, prepare_inpaint_inputs
from eval.methods.base import InpaintingMethod

# SD 2 method for Flickr30K inpainting evaluation.
class SD2InpaintMethod(InpaintingMethod):
    name = "sd2_inpaint"
    model_id = "sd2-community/stable-diffusion-2-inpainting"

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        model_id: str | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
        offload: bool = False,
        negative_prompt: str = "",
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id or self.model_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.offload = offload
        self.negative_prompt = negative_prompt
        self._pipe = None
        self._generator: torch.Generator | None = None

    def load(self) -> None:
        from diffusers import StableDiffusionInpaintPipeline

        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        if self.offload:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(self.device)

        self._generator = make_generator(self.device, self.seed)
        self._loaded = True

    def infer(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        self._ensure_loaded()
        if self._pipe is None:
            raise RuntimeError("SD2 pipeline is not initialized.")

        rgb, mask_l = prepare_inpaint_inputs(image, mask)
        steps = int(kwargs.get("num_inference_steps", self.num_inference_steps))
        guidance = float(kwargs.get("guidance_scale", self.guidance_scale))
        negative = str(kwargs.get("negative_prompt", self.negative_prompt))

        if "seed" in kwargs:
            self._generator = make_generator(self.device, int(kwargs["seed"]))

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            image=rgb,
            mask_image=mask_l,
            height=rgb.height,
            width=rgb.width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=self._generator,
        ).images[0]
        return self._as_rgb(result)

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._generator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()
