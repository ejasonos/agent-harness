from __future__ import annotations

from html.parser import HTMLParser
from html import unescape
import re


class TextExtractor(HTMLParser):
    """
    Extract readable text from HTML while ignoring scripts,
    styles, navigation metadata, and other non-content elements.
    """

    IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
    }

    BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "footer",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "br",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__()

        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs,
    ) -> None:

        tag = tag.lower()

        if tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
            return

        if self._ignored_depth:
            return

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(
        self,
        tag: str,
    ) -> None:

        tag = tag.lower()

        if tag in self.IGNORED_TAGS:
            if self._ignored_depth:
                self._ignored_depth -= 1

            return

        if self._ignored_depth:
            return

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(
        self,
        data: str,
    ) -> None:

        if self._ignored_depth:
            return

        text = unescape(data)

        if text.strip():
            self.parts.append(text)


def _normalize_text(
    text: str,
) -> str:
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text,
    )

    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    return "\n".join(
        line
        for line in lines
        if line
    ).strip()


def extract_text(
    html: str,
    max_length: int = 100_000,
) -> dict[str, object]:
    """
    Extract readable text from an HTML document.
    """

    if not isinstance(html, str):
        raise TypeError(
            "html must be a string."
        )

    if max_length < 1:
        raise ValueError(
            "max_length must be >= 1."
        )

    if max_length > 1_000_000:
        raise ValueError(
            "max_length must be <= 1,000,000."
        )

    parser = TextExtractor()

    parser.feed(html)

    text = _normalize_text(
        "".join(parser.parts)
    )

    truncated = len(text) > max_length

    if truncated:
        text = text[:max_length]

    return {
        "text": text,
        "character_count": len(text),
        "truncated": truncated,
    }


def extract_links(
    html: str,
    max_results: int = 200,
) -> list[dict[str, str]]:
    """
    Extract hyperlinks from an HTML document.
    """

    if max_results < 1:
        raise ValueError(
            "max_results must be >= 1."
        )

    class LinkParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()

            self.links: list[dict[str, str]] = []
            self._href: str | None = None
            self._text: list[str] = []

        def handle_starttag(
            self,
            tag: str,
            attrs,
        ) -> None:

            if tag.lower() != "a":
                return

            attributes = dict(attrs)

            href = attributes.get("href")

            if href:
                self._href = href
                self._text = []

        def handle_data(
            self,
            data: str,
        ) -> None:

            if self._href is not None:
                self._text.append(data)

        def handle_endtag(
            self,
            tag: str,
        ) -> None:

            if tag.lower() != "a":
                return

            if self._href is None:
                return

            text = _normalize_text(
                " ".join(self._text)
            )

            self.links.append(
                {
                    "url": self._href,
                    "text": text,
                }
            )

            self._href = None
            self._text = []

    parser = LinkParser()
    parser.feed(html)

    return parser.links[:max_results]