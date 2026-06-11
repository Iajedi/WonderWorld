"""Run WonderWorld WorldScore benchmarks with BCDM and InternLM text generation.

This entrypoint keeps the WorldScore data flow from ``run_ww_worldscore.py``
while replacing the Stable Diffusion inpaint backend with ``BCDMPipeline``.
``run_ww_worldscore.py`` already imports ``TextpromptGen`` from
``util.internlm`` (OpenAI-compatible interface pointing at the local InternLM
server), so no monkey-patching of the class is required.

BCDM prompt enrichment produces concise ``prompt_src`` / ``prompt_tgt`` pairs
for each inpaint step (vision + text). WorldScore still supplies the raw scene
prompts; ``use_gpt`` (next-scene LLM generation) remains disabled.

Sky point-cloud initialization is handled by
``run_ww_worldscore.bootstrap_sky_pointcloud`` inside ``run()`` (same as the
SD WorldScore runner).
Code adapted from run_ww_worldscore.py using assistance from Cursor Composer 2.5.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser

# ``run_ww_worldscore.py`` imports ``util.chatGPT4`` at module import time.
# BCDM prompt enrichment uses ``util.internlm`` (OpenAI); set a real
# ``OPENAI_API_KEY`` when not passing ``--no_gemini``. The default below only
# satisfies chatGPT4 import-time initialization if nothing is set.
os.environ.setdefault("OPENAI_API_KEY", "unused-for-worldscore-BCDM")


_bcdm_device = "cuda:1"
_bcdm_model = "klein"
_bcdm_offload = False


class _NoOpComponent:
    def set_attn_processor(self, *_args, **_kwargs):
        return None

    def to(self, *_args, **_kwargs):
        return self

    def decode(self, *_args, **_kwargs):
        raise RuntimeError("BCDM adapter does not expose VAE decoding.")


class _NoOpScheduler:
    config = {}

    @classmethod
    def from_config(cls, *_args, **_kwargs):
        return cls()


class _BCDMInpaintAdapter:
    """Adapter matching the small HF inpaint surface used by the old runner."""

    scheduler = _NoOpScheduler()
    unet = _NoOpComponent()
    vae = _NoOpComponent()

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __init__(self):
        from backbone.edit.controller import BCDMPipeline

        self.device = _bcdm_device
        print(
            "Using BCDM inpainting backend "
            f"(model={_bcdm_model}, device={_bcdm_device}, offload={_bcdm_offload})."
        )
        self.pipeline = BCDMPipeline(
            offload=_bcdm_offload,
            model=_bcdm_model,
            device=_bcdm_device,
        )

    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if target is None:
            return self

        target = str(target)
        # The legacy runner sends the inpaint pipeline to config["device"].
        # Keep BCDM on its configured device, except for explicit CPU
        # offloads while SyncDiffusion initializes missing sky panoramas.
        if target == "cpu":
            self.pipeline.to("cpu")
        elif target == self.device:
            self.pipeline.to(self.device)
        return self

    def run(self, *args, **kwargs):
        return self.pipeline.run(*args, **kwargs)


def _install_bcdm_hve_backend(ww_module) -> None:
    ww_module.StableDiffusionInpaintPipeline = _BCDMInpaintAdapter
    ww_module.DDIMScheduler = _NoOpScheduler

# Utilisation of InternLM text generation for BCDM prompt enrichment.
def _install_internlm_text_gen(ww_module) -> None:
    from util.internlm import TextpromptGen as _InternLMTextpromptGen

    ww_module.TextpromptGen = _InternLMTextpromptGen

    original_inpaint = ww_module.KeyframeGen.inpaint

    def inpaint_with_internlm_prompts(self, condition_image, *args, **kwargs):
        # pt_gen is None before run() sets it (sky bootstrap / first frame) -
        # the isinstance guard below naturally skips enrichment in that phase.
        pt_gen = ww_module.pt_gen
        if isinstance(pt_gen, _OpenAITextpromptGen):
            ws_prompt = kwargs.get("inpainting_prompt")
            print("ws_prompt: ", ws_prompt)
            # Skip if caller already supplied explicit BCDM prompts.
            if ws_prompt is not None and "bcdm_prompt_src" not in kwargs:
                ws_prompt_str = ws_prompt if isinstance(ws_prompt, str) else str(ws_prompt)
                scene_dict = {
                    "scene_name": [ws_prompt_str],
                    "entities": [ws_prompt_str],
                    "background": [ws_prompt_str],
                }
                try:
                    bcdm_src, bcdm_tgt = pt_gen.build_bcdm_inpaint_pair_from_conditioning_image(
                        condition_image, ws_prompt.split(" ")[-1], scene_dict, worldscore=True
                    )
                    if bcdm_tgt:
                        # Replace the raw WorldScore prompt with the enriched target.
                        kwargs["inpainting_prompt"] = bcdm_tgt
                        # Inject both sides so the BCDM pipeline uses the full pair
                        # (otherwise models.py falls back to inpainting_prompt for both).
                        kwargs["bcdm_prompt_src"] = bcdm_src
                        kwargs["bcdm_prompt_tgt"] = bcdm_tgt
                except Exception as exc:
                    print(f"[InternLM/OpenAI BCDM] Prompt generation failed, using WorldScore prompt: {exc}")
        return original_inpaint(self, condition_image, *args, **kwargs)

    ww_module.KeyframeGen.inpaint = inpaint_with_internlm_prompts


def _install_layer_prompt_from_current_kf(ww_module) -> None:
    """Use the current WorldScore prompt for layer inpainting.

    ``run_ww_worldscore.py`` calls ``generate_layer(..., scene_name=None)``,
    which makes ``models.py`` fall back to the hard-coded "road, building"
    base-layer prompt.  At that point ``set_kf_param`` has already copied the
    current ``inpainting_prompt_list[i]`` entry into ``self.inpainting_prompt``,
    so use that prompt whenever the caller does not provide an explicit
    ``scene_name``.
    """
    original_generate_layer = ww_module.KeyframeGen.generate_layer

    def generate_layer_with_current_prompt(self, *args, **kwargs):
        if kwargs.get("scene_name") is None:
            kwargs["scene_name"] = self.inpainting_prompt
        return original_generate_layer(self, *args, **kwargs)

    ww_module.KeyframeGen.generate_layer = generate_layer_with_current_prompt


def _prepare_bcdm_config(config, args):
    from omegaconf import OmegaConf

    global _bcdm_device, _bcdm_model, _bcdm_offload

    if args.bcdm_device is not None:
        config.bcdm_device = args.bcdm_device
    elif OmegaConf.select(config, "bcdm_device") is None:
        config.bcdm_device = _bcdm_device

    _bcdm_device = str(config.bcdm_device)
    _bcdm_model = args.bcdm_model
    _bcdm_offload = args.bcdm_offload

    # KeyframeGen selects the BCDM/FLUX mask and prompt handling from this flag.
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
        "--bcdm_device",
        default=None,
        help="Device for BCDM, e.g. cuda:0 or cuda:1. Defaults to config.bcdm_device.",
    )
    parser.add_argument(
        "--bcdm_model",
        default="klein",
        choices=("klein", "flux1"),
        help="BCDM backbone model.",
    )
    parser.add_argument(
        "--bcdm_offload",
        action="store_true",
        help="Enable CPU offload inside BCDM.",
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
            "Scene style hint passed to BCDM prompt generation (OpenAI / internlm; "
            "same flag name as the former Gemini path). "
            "E.g. 'photorealistic', 'oil painting'. Default: photorealistic."
        ),
    )
    parser.add_argument(
        "--no_gemini",
        action="store_true",
        help=(
            "Disable LLM BCDM prompt enrichment (no internlm/OpenAI calls); "
            "use raw WorldScore prompts only."
        ),
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
    # python run_ww_worldscore_bcdm_hve.py --base-config config/base-config.yaml --example_config config/example.yaml --worldscore_model_name wonderworld2 --worldscore_visual_movement static --worldscore_json_file static.json --bcdm_device cuda:1 --bcdm_model klein --bcdm_offload --max_runs 1


def main() -> None:
    args = parse_args()

    from omegaconf import OmegaConf

    import run_ww_worldscore as ww

    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = _prepare_bcdm_config(OmegaConf.merge(base_config, example_config), args)
    config.num_between = args.num_between

    _install_bcdm_hve_backend(ww)
    _install_layer_prompt_from_current_kf(ww)

    if not args.no_gemini:
        _install_gemini_text_gen(ww)

    dataloader, helper = ww.GetHelpers(
        args.worldscore_model_name,
        args.worldscore_visual_movement,
        args.worldscore_json_file,
    )

    shared_inpainter_pipeline = ww.BCDMPipeline(
        offload=_bcdm_offload,
        model=_bcdm_model,
        device=_bcdm_device,
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
