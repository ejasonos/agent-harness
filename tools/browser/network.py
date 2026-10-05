from __future__ import annotations

from typing import Any

from .manager import browser_manager


def browser_get_requests(
    limit: int = 100,
) -> list[dict[str, Any]]:

    if limit < 1:
        raise ValueError(
            "limit must be >= 1."
        )

    return browser_manager.state.requests[
        -limit:
    ]


def browser_get_responses(
    limit: int = 100,
) -> list[dict[str, Any]]:

    if limit < 1:
        raise ValueError(
            "limit must be >= 1."
        )

    return browser_manager.state.responses[
        -limit:
    ]


def browser_clear_network() -> dict[str, Any]:

    browser_manager.clear_network()

    return {
        "cleared": True,
    }