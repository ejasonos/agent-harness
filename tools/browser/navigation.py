from __future__ import annotations

from typing import Any

from .manager import browser_manager


def browser_goto(
    url: str,
    timeout: int = 30_000,
) -> dict[str, Any]:

    url = url.strip()

    if not url:
        raise ValueError(
            "URL cannot be empty."
        )

    if not (
        url.startswith("http://")
        or url.startswith("https://")
    ):
        raise ValueError(
            "Only HTTP and HTTPS URLs are supported."
        )

    page = browser_manager.page()

    response = page.goto(
        url,
        timeout=timeout,
        wait_until="domcontentloaded",
    )

    return {
        "url": page.url,
        "title": page.title(),
        "status": (
            response.status
            if response is not None
            else None
        ),
    }


def browser_back(
    timeout: int = 30_000,
) -> dict[str, Any]:

    page = browser_manager.page()

    response = page.go_back(
        timeout=timeout,
        wait_until="domcontentloaded",
    )

    return {
        "url": page.url,
        "title": page.title(),
        "status": (
            response.status
            if response is not None
            else None
        ),
    }


def browser_forward(
    timeout: int = 30_000,
) -> dict[str, Any]:

    page = browser_manager.page()

    response = page.go_forward(
        timeout=timeout,
        wait_until="domcontentloaded",
    )

    return {
        "url": page.url,
        "title": page.title(),
        "status": (
            response.status
            if response is not None
            else None
        ),
    }


def browser_reload(
    timeout: int = 30_000,
) -> dict[str, Any]:

    page = browser_manager.page()

    response = page.reload(
        timeout=timeout,
        wait_until="domcontentloaded",
    )

    return {
        "url": page.url,
        "title": page.title(),
        "status": (
            response.status
            if response is not None
            else None
        ),
    }


def browser_current_page() -> dict[str, Any]:

    page = browser_manager.page()

    return {
        "url": page.url,
        "title": page.title(),
    }