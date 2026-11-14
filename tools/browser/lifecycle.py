from __future__ import annotations

from typing import Any

from .manager import browser_manager


async def browser_launch(
    headless: bool = True,
    viewport_width: int = 1280,
    viewport_height: int = 900,
) -> dict[str, Any]:

    return await browser_manager.launch(
        headless=headless,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )


async def browser_close() -> dict[str, Any]:

    return await browser_manager.close()


async def browser_status() -> dict[str, Any]:

    return await browser_manager.status()