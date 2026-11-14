from __future__ import annotations

from typing import Any

from .manager import browser_manager


DEFAULT_TIMEOUT = 10_000

_PAGE_SNAPSHOT_SCRIPT = """(body) => {
    const isVisible = (element) => {
        const style = window.getComputedStyle(element);
        return style.display !== "none"
            && style.visibility !== "hidden"
            && element.getClientRects().length > 0;
    };
    const text = (value) => (value || "").replace(/\\s+/g, " ").trim();
    const roleFor = (element) => {
        const explicit = element.getAttribute("role");
        if (explicit) return explicit;
        const tag = element.tagName.toLowerCase();
        if (tag === "a" && element.hasAttribute("href")) return "link";
        if (tag === "button") return "button";
        if (tag === "textarea") return "textbox";
        if (tag === "select") return "combobox";
        if (tag === "input") {
            const type = (element.getAttribute("type") || "text").toLowerCase();
            if (type === "checkbox" || type === "radio") return type;
            if (type === "button" || type === "submit" || type === "reset") {
                return "button";
            }
            if (type === "search") return "searchbox";
            if (type !== "hidden" && type !== "password") return "textbox";
        }
        return null;
    };
    const controls = Array.from(body.querySelectorAll(
        'a[href],button,input:not([type="hidden"]),textarea,select,' +
        '[role="button"],[role="link"],[role="checkbox"],[role="radio"],' +
        '[role="textbox"],[role="combobox"],[contenteditable="true"]'
    )).filter(isVisible);
    const items = controls.slice(0, 200).map((element) => {
        const role = roleFor(element);
        const label = element.labels && element.labels.length
            ? element.labels[0].innerText
            : "";
        const name = text(
            element.getAttribute("aria-label")
            || element.getAttribute("title")
            || label
            || element.innerText
            || element.getAttribute("placeholder")
            || element.getAttribute("alt")
        ).slice(0, 120);
        const selector = role && name
            ? `role=${role}[name=${JSON.stringify(name)}]`
            : (element.id ? `#${CSS.escape(element.id)}` : null);
        return {
            tag: element.tagName.toLowerCase(),
            role,
            name,
            type: element.getAttribute("type"),
            placeholder: element.getAttribute("placeholder"),
            href: element.getAttribute("href"),
            disabled: Boolean(element.disabled),
            selector
        };
    });
    const headings = Array.from(body.querySelectorAll("h1,h2,h3"))
        .filter(isVisible)
        .slice(0, 50)
        .map((heading) => ({
            level: Number(heading.tagName.slice(1)),
            text: text(heading.innerText).slice(0, 200)
        }));
    return {
        headings,
        controls: items,
        control_count: controls.length,
        controls_truncated: controls.length > items.length
    };
}"""


async def browser_page_text(
    max_length: int = 100_000,
) -> dict[str, Any]:

    if max_length < 1:
        raise ValueError(
            "max_length must be >= 1."
        )

    page = browser_manager.page()

    text = await page.locator("body").inner_text()

    truncated = len(text) > max_length

    if truncated:
        text = text[:max_length]

    return {
        "url": page.url,
        "title": await page.title(),
        "text": text,
        "character_count": len(text),
        "truncated": truncated,
    }


async def browser_page_snapshot() -> dict[str, Any]:
    """Return headings and visible controls with role-based locator selectors."""
    page = browser_manager.page()
    snapshot = await page.locator("body").evaluate(_PAGE_SNAPSHOT_SCRIPT)
    return {
        "url": page.url,
        "title": await page.title(),
        **snapshot,
    }


async def browser_element_text(
    selector: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    locator = page.locator(selector)

    text = await locator.inner_text(
        timeout=timeout
    )

    return {
        "selector": selector,
        "text": text,
    }


async def browser_element_attribute(
    selector: str,
    attribute: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:

    page = browser_manager.page()

    value = await page.locator(
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


async def browser_element_exists(
    selector: str,
) -> dict[str, Any]:

    page = browser_manager.page()

    count = await page.locator(
        selector
    ).count()

    return {
        "selector": selector,
        "exists": count > 0,
        "count": count,
    }


async def browser_page_html(
    selector: str | None = None,
    max_length: int = 200_000,
) -> dict[str, Any]:

    page = browser_manager.page()

    if selector is None:
        html = await page.content()
    else:
        html = await page.locator(
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