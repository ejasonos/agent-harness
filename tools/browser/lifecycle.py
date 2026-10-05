from __future__ import annotations

from typing import Any

from .manager import browser_manager


def browser_launch(
    headless: bool = True,
    viewport_width: int = 1280,
    viewport_height: int = 900,
) -> dict[str, Any]:

    return browser_manager.launch(
        headless=headless,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )


def browser_close() -> dict[str, Any]:

    return browser_manager.close()


def browser_status() -> dict[str, Any]:

    return browser_manager.status()