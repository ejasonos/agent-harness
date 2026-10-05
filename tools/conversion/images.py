from __future__ import annotations

from typing import Any

from workspace.manager import WorkspaceManager

from .engine import (
    resolve_input,
    resolve_output,
    require_dependency,
)


def convert_image(
    workspace: WorkspaceManager,
    source_path: str,
    output_path: str,
    output_format: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:

    source = resolve_input(
        workspace,
        source_path,
    )

    destination = resolve_output(
        workspace,
        output_path,
    )

    if width is not None and width < 1:
        raise ValueError(
            "width must be >= 1."
        )

    if height is not None and height < 1:
        raise ValueError(
            "height must be >= 1."
        )

    PIL = require_dependency(
        "PIL",
        "Pillow",
    )

    image = PIL.Image.open(
        str(source)
    )

    original_size = image.size

    if width is not None or height is not None:
        new_width = (
            width
            if width is not None
            else image.width
        )

        new_height = (
            height
            if height is not None
            else image.height
        )

        image = image.resize(
            (
                new_width,
                new_height,
            )
        )

    image_format = (
        output_format
        or destination.suffix.lstrip(".")
    ).upper()

    if image_format == "JPG":
        image_format = "JPEG"

    if image_format in {
        "JPEG",
        "JPG",
    } and image.mode in {
        "RGBA",
        "LA",
        "P",
    }:
        image = image.convert(
            "RGB"
        )

    image.save(
        str(destination),
        format=image_format,
    )

    return {
        "success": True,
        "source": source_path,
        "output": str(destination),
        "format": image_format,
        "original_size": original_size,
        "new_size": image.size,
        "size": destination.stat().st_size,
    }