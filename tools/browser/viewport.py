from __future__ import annotations

from typing import Any

from .manager import browser_manager


def browser_screenshot(
    path: str | None = None,
    full_page: bool = False,
) -> dict[str, Any]:

    page = browser_manager.page()

    kwargs: dict[str, Any] = {
        "full_page": full_page,
    }

    if path is not None:
        kwargs["path"] = path

    data = page.screenshot(
        **kwargs
    )

    return {
        "success": True,
        "url": page.url,
        "path": path,
        "full_page": full_page,
        "size": len(data),
    }


def browser_set_viewport(
    width: int,
    height: int,
) -> dict[str, Any]:

    if width < 320:
        raise ValueError(
            "width must be >= 320."
        )

    if height < 240:
        raise ValueError(
            "height must be >= 240."
        )

    page = browser_manager.page()

    page.set_viewport_size(
        {
            "width": width,
            "height": height,
        }
    )

    return {
        "width": width,
        "height": height,
    }


def browser_scroll(
    x: int = 0,
    y: int = 0,
) -> dict[str, Any]:

    page = browser_manager.page()

    page.mouse.wheel(
        x,
        y,
    )

    return {
        "success": True,
        "x": x,
        "y": y,
    }