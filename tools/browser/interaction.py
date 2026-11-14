from __future__ import annotations

from typing import Any

from .manager import browser_manager


DEFAULT_TIMEOUT = 10_000


async def browser_click(
    selector: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    await page.locator(selector).click(
        timeout=timeout
    )

    return {
        "success": True,
        "action": "click",
        "selector": selector,
        "url": page.url,
    }


async def browser_double_click(
    selector: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    await page.locator(selector).dblclick(
        timeout=timeout
    )

    return {
        "success": True,
        "action": "double_click",
        "selector": selector,
        "url": page.url,
    }


async def browser_fill(
    selector: str,
    value: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    await page.locator(selector).fill(
        value,
        timeout=timeout,
    )

    return {
        "success": True,
        "action": "fill",
        "selector": selector,
    }


async def browser_type(
    selector: str,
    text: str,
    delay: int = 0,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    locator = page.locator(selector)

    await locator.click(
        timeout=timeout
    )

    await locator.press_sequentially(
        text,
        delay=delay,
        timeout=timeout,
    )

    return {
        "success": True,
        "action": "type",
        "selector": selector,
    }


async def browser_press(
    selector: str,
    key: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    await page.locator(selector).press(
        key,
        timeout=timeout,
    )

    return {
        "success": True,
        "action": "press",
        "selector": selector,
        "key": key,
    }


async def browser_select(
    selector: str,
    value: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    selected = await page.locator(selector).select_option(
        value,
        timeout=timeout,
    )

    return {
        "success": True,
        "action": "select",
        "selector": selector,
        "value": value,
        "selected": selected,
    }


async def browser_check(
    selector: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    await page.locator(selector).check(
        timeout=timeout
    )

    return {
        "success": True,
        "action": "check",
        "selector": selector,
    }


async def browser_uncheck(
    selector: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    await page.locator(selector).uncheck(
        timeout=timeout
    )

    return {
        "success": True,
        "action": "uncheck",
        "selector": selector,
    }


async def browser_hover(
    selector: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    await page.locator(selector).hover(
        timeout=timeout
    )

    return {
        "success": True,
        "action": "hover",
        "selector": selector,
    }