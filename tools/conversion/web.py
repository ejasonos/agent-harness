from __future__ import annotations

from typing import Any

from workspace.manager import WorkspaceManager

from .engine import (
    resolve_input,
    resolve_output,
    require_dependency,
)


def html_to_text(
    workspace: WorkspaceManager,
    path: str,
) -> dict[str, Any]:

    source = resolve_input(
        workspace,
        path,
    )

    html = source.read_text(
        encoding="utf-8",
        errors="replace",
    )

    from html.parser import HTMLParser
    import re

    class Parser(HTMLParser):

        def __init__(self) -> None:
            super().__init__()

            self.parts: list[str] = []
            self.ignore_depth = 0

        def handle_starttag(
            self,
            tag,
            attrs,
        ) -> None:

            if tag.lower() in {
                "script",
                "style",
                "noscript",
            }:
                self.ignore_depth += 1

        def handle_endtag(
            self,
            tag,
        ) -> None:

            if tag.lower() in {
                "script",
                "style",
                "noscript",
            } and self.ignore_depth:
                self.ignore_depth -= 1

        def handle_data(
            self,
            data,
        ) -> None:

            if not self.ignore_depth:
                self.parts.append(
                    data
                )

    parser = Parser()
    parser.feed(html)

    text = "".join(
        parser.parts
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return {
        "path": path,
        "text": text,
        "character_count": len(text),
    }


def html_to_pdf(
    workspace: WorkspaceManager,
    source_path: str,
    output_path: str,
) -> dict[str, Any]:

    source = resolve_input(
        workspace,
        source_path,
    )

    destination = resolve_output(
        workspace,
        output_path,
    )

    playwright = require_dependency(
        "playwright",
    )

    from playwright.sync_api import (
        sync_playwright,
    )

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page()

        page.goto(
            source.as_uri(),
            wait_until="networkidle",
        )

        page.pdf(
            path=str(destination),
            format="A4",
            print_background=True,
        )

        browser.close()

    return {
        "success": True,
        "source": source_path,
        "output": output_path,
        "format": "pdf",
        "size": destination.stat().st_size,
    }