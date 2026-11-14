from __future__ import annotations

from typing import Any, Literal

from .manager import browser_manager


BrowserWaitState = Literal[
    "domcontentloaded",
    "load",
    "networkidle",
    "attached",
    "detached",
    "visible",
    "hidden",
]


async def browser_wait(
    selector: str | None = None,
    state: BrowserWaitState = "domcontentloaded",
    timeout: int = 10_000,
) -> dict[str, Any]:
    """Wait for a page load state or for a selector to reach a state."""
    page = browser_manager.page()

    if selector is None:
        if state not in {"domcontentloaded", "load", "networkidle"}:
            raise ValueError(
                "Without a selector, state must be domcontentloaded, load, "
                "or networkidle."
            )
        await page.wait_for_load_state(state, timeout=timeout)
    else:
        if state not in {"attached", "detached", "visible", "hidden"}:
            raise ValueError(
                "With a selector, state must be attached, detached, visible, "
                "or hidden."
            )
        await page.locator(selector).wait_for(
            state=state,
            timeout=timeout,
        )

    return {
        "success": True,
        "url": page.url,
        "selector": selector,
        "state": state,
    }