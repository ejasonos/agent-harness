from __future__ import annotations

from typing import Any

from .manager import browser_manager


DEFAULT_TIMEOUT = 10_000


def browser_page_text(
    max_length: int = 100_000,
) -> dict[str, Any]:

    if max_length < 1:
        raise ValueError(
            "max_length must be >= 1."
        )

    page = browser_manager.page()

    text = page.locator("body").inner_text()

    truncated = len(text) > max_length

    if truncated:
        text = text[:max_length]

    return {
        "url": page.url,
        "title": page.title(),
        "text": text,
        "character_count": len(text),
        "truncated": truncated,
    }


def browser_element_text(
    selector: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    locator = page.locator(selector)

    text = locator.inner_text(
        timeout=timeout
    )

    return {
        "selector": selector,
        "text": text,
    }


def browser_element_attribute(
    selector: str,
    attribute: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    value = page.locator(
        selector
    ).get_attribute(
        attribute,
        timeout=timeout,
    )

    return {
        "selector": selector,
        "attribute": attribute,
        "value": value,
    }


def browser_element_exists(
    selector: str,
) -> dict[str, Any]:

    page = browser_manager.page()

    count = page.locator(
        selector
    ).count()

    return {
        "selector": selector,
        "exists": count > 0,
        "count": count,
    }


def browser_page_html(
    selector: str | None = None,
    max_length: int = 200_000,
) -> dict[str, Any]:

    page = browser_manager.page()

    if selector is None:
        html = page.content()
    else:
        html = page.locator(
            selector
        ).evaluate(
            "(element) => element.outerHTML"
        )

    truncated = len(html) > max_length

    if truncated:
        html = html[:max_length]

    return {
        "selector": selector,
        "html": html,
        "character_count": len(html),
        "truncated": truncated,
    }