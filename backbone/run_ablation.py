#!/usr/bin/env python
"""Run all 5 warm-start variants and produce a comparison grid + metrics.

Usage
-----
    cd backbone
    python run_ablation.py [--config configs/inpaint.yaml] [--outdir outputs/ablation]
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import yaml
from PIL import Image

from backbone.edit.controller import BCOTHVEPipeline
from backbone.viz.debug_transport import save_comparison_grid


WARM_METHODS = [
    "none",
    "kv_replace",
    "masked_attention",
    "ot_only",
    "ot_harmonic",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="BCOT-HVE ablation over warm-start methods")
    parser.add_argument("--config", type=str, default="configs/inpaint.yaml")
    parser.add_argument("--image", type=str, default="inputs/klein_snowman_scaled.png")
    parser.add_argument("--mask", type=str, default="inputs/klein_25p_blur.png")
    parser.add_argument(
        "--prompt_src",
        type=str,
        default=(
            'A cheerful snowman stands on a lakeside promenade in a '
            'Mediterranean-style coastal village, holding a sign that reads '
            '"holl world." Seven pineapples are lined up along the stone ledge '
            'in front of the snowman, with turquoise water and mountainous '
            'terrain in the background.'
        ),
    )
    parser.add_argument(
        "--prompt_tgt",
        type=str,
        default=(
            'A cheerful snowman stands on a lakeside promenade in a '
            'Mediterranean-style coastal village, holding a sign that reads '
            '"holl world." Seven pineapples are lined up along the stone ledge '
            'in front of the snowman, with turquoise water and mountainous '
            'terrain in the background.'
        ),
    )
    parser.add_argument("--outdir", type=str, default="outputs/ablation")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:1", help='Torch device for pipeline, e.g. "cuda:1"')
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    os.makedirs(args.outdir, exist_ok=True)

    # Load mask once for metrics
    original = Image.open(args.image).convert("RGB")
    mask_img = Image.open(args.mask).convert("L").resize((512, 512))
    mask_np = (np.array(mask_img) / 255.0).astype(np.float32).reshape(1, 1, 512, 512)

    pipe = BCOTHVEPipeline(offload=args.offload, device=args.device)

    results: dict[str, Image.Image] = {}
    all_metrics: dict[str, dict] = {}

    for method in WARM_METHODS:
        print(f"\n{'='*60}")
        print(f"[ablation] Running warm_method = {method}")
        print(f"{'='*60}")

        cfg = copy.deepcopy(base_config)
        cfg["warm_method"] = method
        cfg["debug"] = False

        method_dir = os.path.join(args.outdir, method)

        result = pipe.run(
            image=args.image,
            mask=args.mask,
            prompt_src=args.prompt_src,
            prompt_tgt=args.prompt_tgt,
            config=cfg,
            output_dir=method_dir,
        )

        results[method] = result
        metrics = pipe.compute_metrics(original, result, mask_np)
        all_metrics[method] = metrics
        print(f"[ablation] {method}: {metrics}")

    # Save comparison grid
    save_comparison_grid(results, os.path.join(args.outdir, "comparison_grid.png"))

    # Save metrics summary
    with open(os.path.join(args.outdir, "metrics_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("[ablation] Summary")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Preservation':>14} {'Seam Score':>12}")
    print("-" * 48)
    for method in WARM_METHODS:
        m = all_metrics[method]
        print(f"{method:<20} {m['preservation_error']:>14.6f} {m['boundary_seam_score']:>12.6f}")

    print(f"\nComparison grid saved to {os.path.join(args.outdir, 'comparison_grid.png')}")
    print(f"Metrics saved to {os.path.join(args.outdir, 'metrics_summary.json')}")


if __name__ == "__main__":
    main()
