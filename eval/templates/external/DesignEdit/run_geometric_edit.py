"""Thin GeoBench launcher for DesignEdit checkout.

Delegates to wonderworld eval.runners.run_geometric_edit with method=design_edit.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    env_root = os.environ.get("WONDERWORLD_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # eval/external/DesignEdit/run_geometric_edit.py -> repo root
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GeoBench geometric editing with DesignEdit (wonderworld runner)"
    )
    parser.add_argument("--method", type=str, default="design_edit")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--batch-name", type=str, default=None)
    parser.add_argument("--debug-dir", type=str, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--prompt-field",
        type=str,
        default="caption_4v",
        choices=["caption_4v", "edit_prompt"],
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    checkout_root = Path(__file__).resolve().parent

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(checkout_root) not in sys.path:
        sys.path.insert(0, str(checkout_root))

    from eval.paths import resolve_path
    from eval.runners.run_geometric_edit import _resolve_dtype, run_geometric_evaluation

    run_geometric_evaluation(
        method_name=args.method,
        manifest_path=resolve_path(args.manifest, base=repo_root),
        output_dir=resolve_path(args.output_dir, base=repo_root),
        seed=args.seed,
        limit=args.limit,
        sample_id=args.sample_id,
        device=args.device,
        dtype=_resolve_dtype(args.dtype),
        batch_name=args.batch_name,
        prompt_field=args.prompt_field,
        resume=not args.no_resume,
        offload=False,
        debug_dir=args.debug_dir,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
