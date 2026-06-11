"""Path helpers for the evaluation suite."""

from __future__ import annotations

from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
OUTPUT_ROOT = EVAL_ROOT / "output"

# FLICKR30K_1K_DIR = OUTPUT_ROOT / "flickr30k_1k"
# NVIDIA_MASKS_DIR = OUTPUT_ROOT / "nvidia_masks_test"
# MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
# BENCHMARK_MANIFEST = MANIFESTS_DIR / "benchmark_1k.csv"
# ABLATION_OUTPAINT_DIR = MANIFESTS_DIR / "ablation_outpaint"
# ABLATION_OUTPAINT_MANIFEST = MANIFESTS_DIR / "ablation_outpaint.csv"
# GEOBENCH_2D_DIR = OUTPUT_ROOT / "geobench_2d_1k"
# GEOBENCH_2D_MANIFEST = MANIFESTS_DIR / "geobench_2d_1k.csv"
# EXTERNAL_ROOT = EVAL_ROOT / "external"
# DESIGNEDIT_CHECKOUT = EXTERNAL_ROOT / "DesignEdit"
# FREEFINE_CHECKOUT = EXTERNAL_ROOT / "FreeFine"


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a path relative to repo root when not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = base or REPO_ROOT
    return (root / p).resolve()
