"""Dataset-aware image resolution for online multimodal teachers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .fashioniq import resolve_fashioniq_image_path


def resolve_manifest_image_path(
    args: Any, manifest: Mapping[str, Any], image_id: str
) -> Path:
    dataset_name = str(getattr(args, "dataset_name", "fashioniq")).lower()
    if dataset_name == "fashioniq":
        return resolve_fashioniq_image_path(
            args.dataset_root_resolved_images,
            str(image_id),
            str(manifest["category"]),
        )
    raise ValueError(f"Unsupported dataset_name={dataset_name!r}")


__all__ = ["resolve_manifest_image_path"]
