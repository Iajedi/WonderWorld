"""FLUX.1 Fill dev inpainting adapter (Hugging Face)."""

from __future__ import annotations

import torch
from PIL import Image

from eval.methods._hf_common import make_generator, prepare_inpaint_inputs
from eval.methods.base import InpaintingMethod


class FluxFillDevMethod(InpaintingMethod):
    """Text-guided inpainting with ``black-forest-labs/FLUX.1-Fill-dev``."""

    name = "flux_fill_dev"
    model_id = "black-forest-labs/FLUX.1-Fill-dev"

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_id: str | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 30.0,
        max_sequence_length: int = 512,
        seed: int = 42,
        offload: bool = False,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id or self.model_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.max_sequence_length = max_sequence_length
        self.seed = seed
        self.offload = offload
        self._pipe = None
        self._generator: torch.Generator | None = None

    def load(self) -> None:
        from diffusers import FluxFillPipeline

        self._pipe = FluxFillPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
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
            raise RuntimeError("FLUX Fill pipeline is not initialized.")

        rgb, mask_l = prepare_inpaint_inputs(image, mask)
        steps = int(kwargs.get("num_inference_steps", self.num_inference_steps))
        guidance = float(kwargs.get("guidance_scale", self.guidance_scale))
        max_seq = int(kwargs.get("max_sequence_length", self.max_sequence_length))

        if "seed" in kwargs:
            self._generator = make_generator(self.device, int(kwargs["seed"]))

        result = self._pipe(
            prompt=prompt,
            image=rgb,
            mask_image=mask_l,
            height=rgb.height,
            width=rgb.width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            max_sequence_length=max_seq,
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
