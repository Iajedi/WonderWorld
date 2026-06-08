#!/usr/bin/env python
"""Run a single BCDM inpainting example end-to-end.

Usage
-----
    cd /path/to/wonderworld
    python backbone/run_inpaint.py [--config backbone/configs/inpaint.yaml] [--outdir backbone/outputs/bcdm_hve_inpaint]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backbone.edit.controller import BCDMPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="BCDM single inpainting run")
    parser.add_argument("--config", type=str, default="configs/inpaint.yaml")
    parser.add_argument("--image", type=str, default="backbone/inputs/imperial_scaled.jpg")
    parser.add_argument("--mask", type=str, default="backbone/inputs/klein_25p_blur.png")
    parser.add_argument(
        "--prompt_src",
        type=str,
        default="A photo of a red brick building"
    )
    parser.add_argument(
        "--prompt_tgt",
        type=str,
        default="A photo of a red brick building and a dog"
    )
    parser.add_argument("--outdir", type=str, default="backbone/outputs/bcdm_hve_inpaint")
    parser.add_argument("--offload", action="store_true", help="Enable CPU offload")
    parser.add_argument("--device", type=str, default="cuda", help='Torch device for pipeline, e.g. "cuda:1"')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"[run_inpaint] Config: {args.config}")
    print(f"[run_inpaint] warm_method={config.get('warm_method')}, T={config.get('T')}, K={config.get('K')}")

    pipe = BCDMPipeline(offload=args.offload, device=args.device, seed=1)

    # result = pipe.run(
    #     image=args.image,
    #     mask=args.mask,
    #     prompt_src=args.prompt_src,
    #     prompt_tgt=args.prompt_tgt,
    #     config=config,
    #     output_dir=args.outdir,
    # )

    result = pipe.run(
        image=args.image,
        mask=args.mask,
        prompt_src=args.prompt_src,
        prompt_tgt=args.prompt_tgt,
        config=config,
        output_dir=args.outdir,
    )

    original = Image.open(args.image).convert("RGB")
    import numpy as np
    mask_img = Image.open(args.mask).convert("L").resize((512, 512))
    mask_np = (np.array(mask_img) / 255.0).astype(np.float32).reshape(1, 1, 512, 512)
    metrics = pipe.compute_metrics(original, result, mask_np)

    print(f"[run_inpaint] Done. Output in {args.outdir}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
