from __future__ import annotations

from typing import Any

from .manager import browser_manager


def browser_get_console(
    limit: int = 100,
) -> list[dict[str, Any]]:

    if limit < 1:
        raise ValueError(
            "limit must be >= 1."
        )

    return browser_manager.state.console_messages[
        -limit:
    ]


def browser_clear_console() -> dict[str, Any]:

    browser_manager.clear_console()

    return {
        "cleared": True,
    }