## WonderWorld Backbone

Backbone code that integrates into WonderWorld and enables the following:

1. Enables the user to apply fine-grained edits of object addition, modification, and removal in the simulation. This is done through the exploration of geometric editing methods which enable the addition, removal, or replacement of user-specified objects in the scene.
2. Ensures real-time interactivity with the already generated simulation. We demonstrate this by integrating our backbone into WonderWorld while preserving its existing capabilities enabling the backbone process to run asynchronously.
3. Enables training-free, model-agnostic adoption of base models for inpainting, out-painting and geometric editing tasks. We demonstrate this by implementing our method on FLUX.2 [klein] 4B `Flux2KleinPipeline`, a state-of-the-art image generation model not purposed for the aforementioned tasks

## Backbone algorithm

Our backbone utilises the following steps:

1. Inversion with Uni-Inv
2. Optimal transport step to transport known velocities to unknown region
3. Poisson interpolation (harmonic extension) step to interpolate smoothly between known and unknown region
4. Guidance mask modification to use inpainting mask during Uni-Edit
5. Reinjection of observed tokens during denoising for colorimetric consistency

In particular, steps 2 and 3 will be skipped for inpainting-only tasks; since they enable use of UniEdit-Flow for outpainting.

## Geometric editing

1. Given an input image `I` containing an object to be edited, we allow the user to define a binary mask `M` corresponding to the object. Such a mask could be manually drawn or derived from SAM. The user also supplies an editing instruction (from Photoshop-like handles) in which we denote as an affine transformation function `T`.
2. We inpaint the region defined by mask `M` using our method, yielding image `I_bg`. Since UniEdit uses prompt conditioning, we utilise an LLM to encode a prompt describing `I` *without* contents in $M$, and utilise it as the source and target prompts. The system prompts for the LLM are given in `../util/internlm.py`. Moreover, UniEdit-Flow uses the existing latent code within `M` to fill the region, hence it would ideally result in natural blending with surrounding context.
3. We apply transformation function `T` on `I` and `M` to transform the object and mask according to user instructions, and compose the masked object onto background image `I_0` to yield composed image `I_c`.

## Backbone entrypoint

Our backbone can be called independently by changing `__main__` code with input images and prompts

```python
cd backbone
python pipeline.py
```

