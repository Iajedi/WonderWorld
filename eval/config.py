"""Default configuration for the inpainting evaluation suite."""

from __future__ import annotations

import os
from pathlib import Path

_EVAL_ROOT = Path(__file__).resolve().parent

# Flickr30k
DEFAULT_FLICKR30K_DATASET = "nlphuji/flickr30k"
STREAMING_FLICKR30K_DATASET = "lmms-lab/flickr30k"
DEFAULT_FLICKR30K_SPLIT = "test"
DEFAULT_SUBSET_COUNT = 1000
DEFAULT_SUBSET_SEED = 42
DEFAULT_CAPTION_STRATEGY = "first"  # "first" | "seeded_random"

# NVIDIA irregular mask testing set
# Override via CLI --url if this link is stale.
# Official page: https://nv-adlr.github.io/publication/partialconv-inpainting
# Official testing-set archive (Dropbox mirror linked from NVIDIA ADLR page).
NVIDIA_TEST_MASK_ZIP_URL = (
    "https://www.dropbox.com/s/01dfayns9s0kevy/test_mask.zip?dl=1"
)
NVIDIA_OFFICIAL_PAGE_URL = "https://nv-adlr.github.io/publication/partialconv-inpainting"
DEFAULT_MASK_COUNT = 1000
DEFAULT_MASK_BINARIZE_THRESHOLD = 127

# NVIDIA testing set layout (12k masks in test_mask.zip):
# 6 hole-to-image area bins x 2000 masks (1000 without border, 1000 with border).
NVIDIA_MASKS_PER_BIN = 2000
NVIDIA_MASKS_PER_BIN_VARIANT = 1000
NVIDIA_AREA_BINS: tuple[tuple[float, float], ...] = (
    (0.01, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
)
# Default benchmark bins: keep holes small so most of the image remains visible.
DEFAULT_NVIDIA_BIN_INDICES: tuple[int, ...] = (0, 1, 2)

# Benchmark geometry
DEFAULT_TARGET_SIZE = 512
DEFAULT_PAIRING_MODE = "fixed"  # "fixed" | "random"

# Right-side outpaint ablation (processed pano scenes)
OUTPAINT_COVERAGE_LEVELS: tuple[float, ...] = (
    0.05,
    0.15,
    0.25,
    0.35,
    0.45,
    0.55,
    0.65,
    0.75,
    0.85,
    0.95,
)
DEFAULT_OUTPAINT_BLUR_SIGMA = 10.0
DEFAULT_PANO_MANIFEST = "eval/output/manifests/processed_pano/manifest.csv"
DEFAULT_PANO_IMAGES_DIR = "eval/output/manifests/processed_pano/images"
DEFAULT_ABLATION_OUTPAINT_DIR = "eval/output/manifests/ablation_outpaint"
DEFAULT_ABLATION_OUTPAINT_CSV = "eval/output/manifests/ablation_outpaint.csv"
DEFAULT_WORLDSCORE_STATIC_ROOT = "worldscore_output_flux"

# Metrics
DEFAULT_MASKED_METRICS = True
# open_clip matches benchmark CLIP scores (~0.3); pyiqa clipscore uses a different scale (~0.8).
CLIP_BACKEND = os.environ.get("EVAL_CLIP_BACKEND", "open_clip")
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
LPIPS_NET = "alex"

# GeoBench 2D geometric editing
DEFAULT_GEOBENCH_DATASET = "CIawevy/GeoBench"
DEFAULT_GEOBENCH_CONFIG = "2d"
DEFAULT_GEOBENCH_SPLIT = "data"
DEFAULT_GEOBENCH_COUNT = 1000
DEFAULT_GEOBENCH_OUT = "eval/output/geobench_2d_1k"
DEFAULT_GEOBENCH_MANIFEST = "eval/output/manifests/geobench_2d_1k.csv"
DEFAULT_GEOBENCH_PROMPT_FIELD = "caption_4v"  # "caption_4v" | "edit_prompt"

# Runner
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "float16"

# External geometric-editing baselines (cloned under eval/external/)
DESIGNEDIT_CHECKOUT = _EVAL_ROOT / "external" / "DesignEdit"
FREEFINE_CHECKOUT = _EVAL_ROOT / "external" / "FreeFine"
DESIGNEDIT_SDXL_PATH = os.environ.get(
    "DESIGNEDIT_SDXL_PATH", "stabilityai/stable-diffusion-xl-base-1.0"
)
FREEFINE_SD15_PATH = os.environ.get(
    "FREEFINE_SD15_PATH", "runwayml/stable-diffusion-v1-5"
)
DESIGNEDIT_CANVAS_SIZE = 512
FREEFINE_CANVAS_SIZE = 512
FREEFINE_MASK_DILATION = 30
