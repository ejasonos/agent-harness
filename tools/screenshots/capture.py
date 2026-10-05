from __future__ import annotations

from pathlib import Path
from typing import Any

from workspace.manager import WorkspaceManager

from tools.conversion.engine import (
    resolve_input,
    resolve_output,
    require_dependency,
)


def capture_screen(
    workspace: WorkspaceManager,
    output_path: str,
) -> dict[str, Any]:
    """
    Capture the primary screen and save it into the workspace.
    """

    destination = resolve_output(
        workspace,
        output_path,
    )

    pyautogui = require_dependency(
        "pyautogui",
    )

    try:
        image = pyautogui.screenshot()

        image.save(
            str(destination)
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to capture screen: {exc}"
        ) from exc

    return {
        "success": True,
        "path": str(destination),
        "width": image.width,
        "height": image.height,
        "format": destination.suffix.lstrip(".").lower(),
        "size": destination.stat().st_size,
    }


def capture_region(
    workspace: WorkspaceManager,
    output_path: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """
    Capture a rectangular region of the primary screen.
    """

    if width <= 0:
        raise ValueError(
            "width must be > 0."
        )

    if height <= 0:
        raise ValueError(
            "height must be > 0."
        )

    destination = resolve_output(
        workspace,
        output_path,
    )

    pyautogui = require_dependency(
        "pyautogui",
    )

    try:
        image = pyautogui.screenshot(
            region=(
                x,
                y,
                width,
                height,
            )
        )

        image.save(
            str(destination)
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to capture screen region: {exc}"
        ) from exc

    return {
        "success": True,
        "path": str(destination),
        "x": x,
        "y": y,
        "width": image.width,
        "height": image.height,
        "size": destination.stat().st_size,
    }


def image_info(
    workspace: WorkspaceManager,
    path: str,
) -> dict[str, Any]:
    """
    Inspect image metadata.
    """

    source = resolve_input(
        workspace,
        path,
    )

    PIL = require_dependency(
        "PIL",
        "Pillow",
    )

    try:
        image = PIL.Image.open(
            str(source)
        )

        return {
            "path": path,
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "size": source.stat().st_size,
        }

    except Exception as exc:
        raise RuntimeError(
            f"Failed to inspect image: {exc}"
        ) from exc


def crop_image(
    workspace: WorkspaceManager,
    source_path: str,
    output_path: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> dict[str, Any]:
    """
    Crop an image to the specified rectangle.
    """

    if right <= left:
        raise ValueError(
            "right must be greater than left."
        )

    if bottom <= top:
        raise ValueError(
            "bottom must be greater than top."
        )

    source = resolve_input(
        workspace,
        source_path,
    )

    destination = resolve_output(
        workspace,
        output_path,
    )

    PIL = require_dependency(
        "PIL",
        "Pillow",
    )

    try:
        image = PIL.Image.open(
            str(source)
        )

        cropped = image.crop(
            (
                left,
                top,
                right,
                bottom,
            )
        )

        cropped.save(
            str(destination)
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to crop image: {exc}"
        ) from exc

    return {
        "success": True,
        "source": source_path,
        "output": output_path,
        "width": cropped.width,
        "height": cropped.height,
        "size": destination.stat().st_size,
    }


def resize_image(
    workspace: WorkspaceManager,
    source_path: str,
    output_path: str,
    width: int,
    height: int | None = None,
) -> dict[str, Any]:
    """
    Resize an image.

    If height is omitted, preserve the original aspect ratio.
    """

    if width <= 0:
        raise ValueError(
            "width must be > 0."
        )

    if height is not None and height <= 0:
        raise ValueError(
            "height must be > 0."
        )

    source = resolve_input(
        workspace,
        source_path,
    )

    destination = resolve_output(
        workspace,
        output_path,
    )

    PIL = require_dependency(
        "PIL",
        "Pillow",
    )

    try:
        image = PIL.Image.open(
            str(source)
        )

        if height is None:
            ratio = width / image.width

            height = max(
                1,
                round(
                    image.height * ratio
                ),
            )

        resized = image.resize(
            (
                width,
                height,
            ),
            PIL.Image.Resampling.LANCZOS,
        )

        resized.save(
            str(destination)
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to resize image: {exc}"
        ) from exc

    return {
        "success": True,
        "source": source_path,
        "output": output_path,
        "width": resized.width,
        "height": resized.height,
        "size": destination.stat().st_size,
    }