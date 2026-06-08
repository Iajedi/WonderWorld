"""Manifest dataclasses and prompt selection helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FlickrRecord:
    sample_id: str
    split: str
    dataset_index: int
    img_id: str
    filename: str
    image_path: str
    chosen_prompt: str
    all_captions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["all_captions"] = self.all_captions
        return d


@dataclass
class MaskRecord:
    mask_id: str
    source_url: str
    source_member: str
    mask_path: str
    original_width: int
    original_height: int
    mask_area_ratio: float
    area_bin: str = ""
    border_constraint: bool = False
    source_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSample:
    sample_id: str
    image_path: str
    prompt: str
    all_captions_json: str
    mask_id: str
    mask_path: str
    target_width: int
    target_height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AblationOutpaintSample:
    """One row in the right-side outpaint ablation manifest."""

    sample_id: str
    image_path: str
    composed_path: str
    prompt: str
    all_captions_json: str
    mask_id: str
    mask_path: str
    target_width: int
    target_height: int
    scene_id: str
    outpaint_coverage: float
    prompt_src: str = ""
    prompt_tgt: str = ""
    prompt_variant: str = "first"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ABLATION_OUTPAINT_FIELDS = [
    "sample_id",
    "image_path",
    "composed_path",
    "prompt",
    "all_captions_json",
    "mask_id",
    "mask_path",
    "target_width",
    "target_height",
    "scene_id",
    "outpaint_coverage",
    "prompt_src",
    "prompt_tgt",
    "prompt_variant",
]


@dataclass
class GeoBenchRecord:
    """One row from the GeoBench 2D subset export."""

    sample_id: str
    dataset_name: str
    config_name: str
    split_name: str
    dataset_index: int
    ori_img_path: str
    ori_mask_path: str
    tgt_mask_path: str
    caption_4v: str
    edit_prompt: str = ""
    edit_param_json: str = ""
    edit_dx: float = 0.0
    edit_dy: float = 0.0
    edit_dz: float = 0.0
    edit_rx: float = 0.0
    edit_ry: float = 0.0
    edit_rz: float = 0.0
    edit_sx: float = 1.0
    edit_sy: float = 1.0
    edit_sz: float = 1.0
    obj_label: str = ""
    coarse_input_path: str = ""
    original_width: int = 0
    original_height: int = 0
    mask_width: int = 0
    mask_height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeoBenchBenchmarkSample:
    """One row in the GeoBench geometric-editing benchmark manifest."""

    sample_id: str
    ori_img_path: str
    ori_mask_path: str
    tgt_mask_path: str
    prompt: str
    caption_4v: str
    edit_prompt: str
    edit_param_json: str
    obj_label: str
    target_width: int
    target_height: int
    coarse_input_path: str = ""
    edit_param_iou: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GEOBENCH_SUBSET_FIELDS = [
    "sample_id",
    "dataset_name",
    "config_name",
    "split_name",
    "dataset_index",
    "ori_img_path",
    "ori_mask_path",
    "tgt_mask_path",
    "coarse_input_path",
    "caption_4v",
    "edit_prompt",
    "edit_param_json",
    "edit_dx",
    "edit_dy",
    "edit_dz",
    "edit_rx",
    "edit_ry",
    "edit_rz",
    "edit_sx",
    "edit_sy",
    "edit_sz",
    "obj_label",
    "original_width",
    "original_height",
    "mask_width",
    "mask_height",
]

GEOBENCH_MANIFEST_FIELDS = [
    "sample_id",
    "ori_img_path",
    "ori_mask_path",
    "tgt_mask_path",
    "prompt",
    "caption_4v",
    "edit_prompt",
    "edit_param_json",
    "obj_label",
    "coarse_input_path",
    "edit_param_iou",
    "target_width",
    "target_height",
]


def choose_caption(
    captions: list[str],
    *,
    sample_id: str,
    strategy: str = "first",
    seed: int = 42,
) -> str:
    """Select one caption deterministically from a list."""
    if not captions:
        raise ValueError("No captions available for prompt selection.")
    if strategy == "first":
        return captions[0]
    if strategy == "seeded_random":
        idx = hash((sample_id, seed)) % len(captions)
        return captions[idx]
    raise ValueError(f"Unknown caption strategy: {strategy}")


def captions_to_json(captions: list[str]) -> str:
    """Serialize captions list to JSON string."""
    return json.dumps(captions, ensure_ascii=False)


def captions_from_json(raw: str) -> list[str]:
    """Deserialize captions JSON string."""
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("all_captions_json must decode to a list.")
    return [str(x) for x in data]
