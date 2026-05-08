"""Socket.IO edit payload helpers for Wonderworld geometry edits."""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

try:
    from ..geometry.spec import GeometrySpec
except ImportError:
    from backbone.geometry.spec import GeometrySpec


_CANVAS_SIZE = (512, 512)
_REQUIRED_KEYS = ("edit_type", "source_image", "source_mask", "target_image", "target_mask")
_VALID_EDIT_TYPES = {"manipulation", "copy", "addition", "replacement"}


class EditPayloadError(ValueError):
    """Raised when an edit_submit payload is missing fields or has invalid images."""


@dataclass(frozen=True)
class DecodedEditPayload:
    edit_type: str
    source_image: Image.Image
    source_mask: Image.Image
    target_image: Image.Image
    target_mask: Image.Image
    source_mask_tensor: torch.Tensor
    target_mask_tensor: torch.Tensor
    removes_source_region: bool


def _strip_data_url(payload: str) -> str:
    value = payload.strip()
    if value.startswith("data:"):
        header, sep, data = value.partition(",")
        if not sep:
            raise EditPayloadError("Image data URL is missing a comma separator.")
        if not header.lower().startswith("data:image/png"):
            raise EditPayloadError("Only PNG image data URLs are supported.")
        if ";base64" not in header.lower():
            raise EditPayloadError("Only base64 image data URLs are supported.")
        return data.strip()
    return value


def _decode_image_payload(value: Any, field_name: str) -> Image.Image:
    if not isinstance(value, str) or not value.strip():
        raise EditPayloadError(f"'{field_name}' must be a non-empty base64 PNG payload.")

    try:
        raw = base64.b64decode(_strip_data_url(value), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EditPayloadError(f"'{field_name}' is not valid base64 image data.") from exc

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image_format = image.format
            image.load()
            decoded = image.copy()
    except Exception as exc:  # noqa: BLE001 - PIL raises several image-specific exceptions.
        raise EditPayloadError(f"'{field_name}' could not be decoded as an image.") from exc

    if image_format != "PNG":
        raise EditPayloadError(f"'{field_name}' must be PNG image data.")
    if decoded.size != _CANVAS_SIZE:
        raise EditPayloadError(f"'{field_name}' must be 512x512, got {decoded.size[0]}x{decoded.size[1]}.")
    return decoded


def _normalize_mask(image: Image.Image, field_name: str) -> Image.Image:
    mask = image.convert("L")
    arr = np.asarray(mask, dtype=np.uint8)
    binary = np.where(arr >= 128, 255, 0).astype(np.uint8)
    if binary.shape != (_CANVAS_SIZE[1], _CANVAS_SIZE[0]):
        raise EditPayloadError(f"'{field_name}' must be a 512x512 mask.")
    return Image.fromarray(binary, mode="L")


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def decode_edit_payload(data: Any) -> DecodedEditPayload:
    """Validate and decode a frontend ``edit_submit`` payload."""

    if not isinstance(data, Mapping):
        raise EditPayloadError("Edit payload must be an object.")

    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise EditPayloadError(f"Edit payload is missing required key(s): {', '.join(missing)}.")

    edit_type = str(data["edit_type"]).strip().lower()
    if edit_type not in _VALID_EDIT_TYPES:
        valid = ", ".join(sorted(_VALID_EDIT_TYPES))
        raise EditPayloadError(f"'edit_type' must be one of: {valid}.")

    source_image = _decode_image_payload(data["source_image"], "source_image").convert("RGB")
    target_image = _decode_image_payload(data["target_image"], "target_image").convert("RGB")
    source_mask = _normalize_mask(_decode_image_payload(data["source_mask"], "source_mask"), "source_mask")
    target_mask = _normalize_mask(_decode_image_payload(data["target_mask"], "target_mask"), "target_mask")

    return DecodedEditPayload(
        edit_type=edit_type,
        source_image=source_image,
        source_mask=source_mask,
        target_image=target_image,
        target_mask=target_mask,
        source_mask_tensor=_mask_to_tensor(source_mask),
        target_mask_tensor=_mask_to_tensor(target_mask),
        removes_source_region=edit_type in {"manipulation", "replacement"},
    )


def masked_source_for_caption(source_image: Image.Image, source_mask: Image.Image) -> Image.Image:
    """Return the source image with the original selected region blacked out."""

    source = source_image.convert("RGB")
    mask = _normalize_mask(source_mask, "source_mask")
    blank = Image.new("RGB", source.size, (0, 0, 0))
    return Image.composite(blank, source, mask)


def build_socket_geometry_spec(
    decoded: DecodedEditPayload,
    prompt_inpaint: str,
    prompt_refine: str | None,
) -> GeometrySpec:
    """Build an edit spec that preserves frontend-provided 512x512 alignment."""

    identity = torch.eye(3, dtype=torch.float32)
    removal_mask = (
        torch.maximum(decoded.source_mask_tensor, decoded.target_mask_tensor)
        if decoded.removes_source_region
        else decoded.target_mask_tensor
    )
    return GeometrySpec.for_compose_multi(
        removal_masks=[removal_mask],
        compose_layers=[(identity, decoded.target_mask_tensor)],
        prompt_inpaint=prompt_inpaint,
        prompt_refine=prompt_refine or prompt_inpaint,
        mask_user=None,
    )


def image_to_png_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a PNG data URL for direct frontend display."""

    with io.BytesIO() as buffer:
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

