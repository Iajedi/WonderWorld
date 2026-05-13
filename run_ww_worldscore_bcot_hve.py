"""Run WonderWorld WorldScore benchmarks with BCOT-HVE and Gemini text generation enabled.

This entrypoint keeps the WorldScore data flow from ``run_ww_worldscore.py``
while replacing the Stable Diffusion inpaint backend with ``BCOTHVEPipeline``
and the ChatGPT4 prompt generator with ``GeminiTextpromptGen``.

Gemini text generation produces concise BCOT src/tgt prompt pairs for each
inpaint step: ``prompt_src`` describes the conditioning frame (vision call) and
``prompt_tgt`` extends it seamlessly into the WorldScore target scene (text
call).  WorldScore still supplies the raw scene prompts; ``use_gpt`` (next-scene
LLM generation) remains disabled.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser

# ``run_ww_worldscore.py`` imports ``util.chatGPT4`` at module import time.
# The benchmark path does not use that client, but the OpenAI SDK still expects
# an API key to exist when the module-level client is constructed.
os.environ.setdefault("OPENAI_API_KEY", "unused-for-worldscore-bcot-hve")


_bcot_device = "cuda:1"
_bcot_model = "klein"
_bcot_offload = False


class _NoOpComponent:
    def set_attn_processor(self, *_args, **_kwargs):
        return None

    def to(self, *_args, **_kwargs):
        return self

    def decode(self, *_args, **_kwargs):
        raise RuntimeError("BCOT-HVE adapter does not expose VAE decoding.")


class _NoOpScheduler:
    config = {}

    @classmethod
    def from_config(cls, *_args, **_kwargs):
        return cls()


class _BCOTHVEInpaintAdapter:
    """Adapter matching the small HF inpaint surface used by the old runner."""

    scheduler = _NoOpScheduler()
    unet = _NoOpComponent()
    vae = _NoOpComponent()

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __init__(self):
        from backbone.edit.controller import BCOTHVEPipeline

        self.device = _bcot_device
        print(
            "Using BCOT-HVE inpainting backend "
            f"(model={_bcot_model}, device={_bcot_device}, offload={_bcot_offload})."
        )
        self.pipeline = BCOTHVEPipeline(
            offload=_bcot_offload,
            model=_bcot_model,
            device=_bcot_device,
        )

    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if target is None:
            return self

        target = str(target)
        # The legacy runner sends the inpaint pipeline to config["device"].
        # Keep BCOT-HVE on its configured device, except for explicit CPU
        # offloads while SyncDiffusion initializes missing sky panoramas.
        if target == "cpu":
            self.pipeline.to("cpu")
        elif target == self.device:
            self.pipeline.to(self.device)
        return self

    def run(self, *args, **kwargs):
        return self.pipeline.run(*args, **kwargs)


def _install_bcot_hve_backend(ww_module) -> None:
    ww_module.StableDiffusionInpaintPipeline = _BCOTHVEInpaintAdapter
    ww_module.DDIMScheduler = _NoOpScheduler


def _install_gemini_text_gen(ww_module, style: str = "photorealistic") -> None:
    """Replace TextpromptGen with GeminiTextpromptGen and wire BCOT prompt generation.

    Two changes are made to ``ww_module``:

    1. ``ww_module.TextpromptGen`` is replaced with ``GeminiTextpromptGen`` so
       that ``ww_module.run()`` instantiates a Gemini-backed ``pt_gen``.

    2. ``KeyframeGen.inpaint`` is wrapped to call
       ``pt_gen.build_bcot_inpaint_pair_from_conditioning_image`` before each
       inpaint step.  WorldScore still supplies the raw scene prompts.

    **First image**: Gemini is not applied.  ``ww_module.pt_gen`` is ``None``
    until ``run()`` sets it at line 172 (after the first-frame point-cloud and
    sky bootstrap at line 169), so any inpaint that happens during sky
    initialisation is passed through unchanged.

    **Subsequent images**: once ``pt_gen`` is a ``GeminiTextpromptGen``
    instance, each inpaint call in the main scene loop gets Gemini enrichment:
    the raw WorldScore prompt becomes the target-scene context for a two-call
    Gemini pipeline (vision ``bcot_src`` + text ``bcot_tgt``).  Both
    ``bcot_prompt_src`` and ``bcot_prompt_tgt`` are injected into the inpaint
    kwargs so the BCOT pipeline uses the proper prompt pair rather than
    falling back to ``self.inpainting_prompt`` for both sides.  On failure the
    original WorldScore prompt is used unchanged.
    """
    from util.gemini_prompt_gen import GeminiTextpromptGen

    ww_module.TextpromptGen = GeminiTextpromptGen

    original_inpaint = ww_module.KeyframeGen.inpaint

    def inpaint_with_gemini_prompts(self, condition_image, *args, **kwargs):
        # pt_gen is None before run() sets it (sky bootstrap / first frame) —
        # the isinstance guard below naturally skips Gemini in that phase.
        pt_gen = ww_module.pt_gen
        if isinstance(pt_gen, GeminiTextpromptGen):
            ws_prompt = kwargs.get("inpainting_prompt")
            # Skip if caller already supplied explicit BCOT prompts.
            if ws_prompt is not None and "bcot_prompt_src" not in kwargs:
                ws_prompt_str = ws_prompt if isinstance(ws_prompt, str) else str(ws_prompt)
                scene_dict = {
                    "scene_name": [ws_prompt_str],
                    "entities": [ws_prompt_str],
                    "background": [ws_prompt_str],
                }
                try:
                    bcot_src, bcot_tgt = pt_gen.build_bcot_inpaint_pair_from_conditioning_image(
                        condition_image, style, scene_dict
                    )
                    if bcot_tgt:
                        # Replace the raw WorldScore prompt with the enriched target.
                        kwargs["inpainting_prompt"] = bcot_tgt
                        # Inject both sides so the BCOT pipeline uses the full pair
                        # (otherwise models.py falls back to inpainting_prompt for both).
                        kwargs["bcot_prompt_src"] = bcot_src
                        kwargs["bcot_prompt_tgt"] = bcot_tgt
                except Exception as exc:
                    print(f"[GeminiTextGen] BCOT prompt generation failed, using WorldScore prompt: {exc}")
        return original_inpaint(self, condition_image, *args, **kwargs)

    ww_module.KeyframeGen.inpaint = inpaint_with_gemini_prompts


def _install_sky_bootstrap(ww_module) -> None:
    original_recompose = ww_module.KeyframeGen.recompose_image_latest_and_set_current_pc

    def recompose_with_sky_bootstrap(self, *args, **kwargs):
        if self.current_pc_sky is None:
            example = self.config["example_name"]
            gen_sky_image = bool(self.config["gen_sky_image"])
            sky_0 = f"./examples/sky_images/{example}/sky_0.png"
            sky_1 = f"./examples/sky_images/{example}/sky_1.png"
            needs_syncdiffusion = gen_sky_image or not (os.path.exists(sky_0) and os.path.exists(sky_1))
            syncdiffusion_model = None
            inpainter_home_device = (
                getattr(self.inpainting_pipeline, "device", None)
                if needs_syncdiffusion
                else None
            )
            if needs_syncdiffusion and hasattr(self.inpainting_pipeline, "to"):
                self.inpainting_pipeline.to("cpu")
                ww_module.empty_cache()

            try:
                if needs_syncdiffusion:
                    from syncdiffusion.syncdiffusion_model import SyncDiffusion

                    sync_device = self.config["bcot_device"] if self.config["use_flux"] else self.config["device"]
                    syncdiffusion_model = SyncDiffusion(sync_device, sd_version="2.0-inpaint")

                sky_mask = self.generate_sky_mask().float()
                if not sky_mask.bool().any().item():
                    print(
                        "[WARN] No sky pixels found in the WorldScore start frame; "
                        "using the top image band to initialize the sky point cloud."
                    )
                    sky_mask = sky_mask.clone()
                    sky_mask[:128, :] = 1.0
                self.generate_sky_pointcloud(
                    syncdiffusion_model,
                    image=self.image_latest,
                    mask=sky_mask,
                    gen_sky=gen_sky_image,
                    style=None,
                )
            finally:
                syncdiffusion_model = None
                ww_module.empty_cache()
                if inpainter_home_device is not None and hasattr(self.inpainting_pipeline, "to"):
                    self.inpainting_pipeline.to(inpainter_home_device)

        return original_recompose(self, *args, **kwargs)

    ww_module.KeyframeGen.recompose_image_latest_and_set_current_pc = recompose_with_sky_bootstrap


def _prepare_bcot_config(config, args):
    from omegaconf import OmegaConf

    global _bcot_device, _bcot_model, _bcot_offload

    if args.bcot_device is not None:
        config.bcot_device = args.bcot_device
    elif OmegaConf.select(config, "bcot_device") is None:
        config.bcot_device = _bcot_device

    _bcot_device = str(config.bcot_device)
    _bcot_model = args.bcot_model
    _bcot_offload = args.bcot_offload

    # KeyframeGen selects the BCOT/FLUX mask and prompt handling from this flag.
    config.use_flux = True

    # WorldScore supplies all prompts. Leaving this enabled in the legacy runner
    # can dereference scene_dict before it is initialized.
    config.use_gpt = False

    # The adapter intentionally has no UNet/VAE objects to compile.
    config.use_compile = False
    return config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config",
        default="./config/example.yaml",
    )
    parser.add_argument(
        "--worldscore_visual_movement",
        default="static",
    )
    parser.add_argument(
        "--worldscore_json_file",
        default="static.json",
        help="json file name",
    )
    parser.add_argument(
        "--worldscore_model_name",
        default="wonderworld",
        help="Model name",
    )
    parser.add_argument(
        "--bcot_device",
        default=None,
        help="Device for BCOT-HVE, e.g. cuda:0 or cuda:1. Defaults to config.bcot_device.",
    )
    parser.add_argument(
        "--bcot_model",
        default="klein",
        choices=("klein", "flux1"),
        help="BCOT-HVE backbone model.",
    )
    parser.add_argument(
        "--bcot_offload",
        action="store_true",
        help="Enable CPU offload inside BCOT-HVE.",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Maximum WorldScore samples to run. Defaults to all; use 0 for setup-only smoke checks.",
    )
    parser.add_argument(
        "--gemini_style",
        default="photorealistic",
        help=(
            "Scene style hint passed to Gemini BCOT prompt generation "
            "(e.g. 'photorealistic', 'oil painting'). Default: photorealistic."
        ),
    )
    parser.add_argument(
        "--no_gemini",
        action="store_true",
        help="Disable Gemini text generation and fall back to the raw WorldScore prompts.",
    )
    parser.add_argument(
        "--num_between",
        type=int,
        default=0,
        help="Number of interpolated in-between scenes per keyframe segment. Default: 0.",
    )
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs < 0:
        parser.error("--max_runs must be non-negative")
    return args

    # Example run
    # python run_ww_worldscore_bcot_hve.py --base-config config/base-config.yaml --example_config config/example.yaml --worldscore_model_name wonderworld2 --worldscore_visual_movement static --worldscore_json_file static.json --bcot_device cuda:1 --bcot_model klein --bcot_offload --max_runs 1


def main() -> None:
    args = parse_args()

    from omegaconf import OmegaConf

    import run_ww_worldscore as ww

    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = _prepare_bcot_config(OmegaConf.merge(base_config, example_config), args)
    config.num_between = args.num_between

    _install_bcot_hve_backend(ww)
    _install_sky_bootstrap(ww)

    if not args.no_gemini:
        _install_gemini_text_gen(ww, style=args.gemini_style)

    dataloader, helper = ww.GetHelpers(
        args.worldscore_model_name,
        args.worldscore_visual_movement,
        args.worldscore_json_file,
    )

    shared_inpainter_pipeline = ww.BCOTHVEPipeline(
        offload=_bcot_offload,
        model=_bcot_model,
        device=_bcot_device,
    )

    for run_idx, data in enumerate(dataloader):
        if args.max_runs is not None and run_idx >= args.max_runs:
            break
        start_keyframe, inpainting_prompt_list, cameras, cameras_interp = helper.adapt(data)
        config.num_scenes = data["num_scenes"]
        ww.run(
            config,
            start_keyframe,
            inpainting_prompt_list,
            cameras,
            cameras_interp,
            helper,
            inpainter_pipeline=shared_inpainter_pipeline,
        )
        ww.empty_cache()


if __name__ == "__main__":
    main()
