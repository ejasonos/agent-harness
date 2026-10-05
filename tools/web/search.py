from __future__ import annotations

from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from html.parser import HTMLParser


DEFAULT_SEARCH_ENGINE = "https://www.google.com/search?q="


class SearchResultParser(HTMLParser):
    """
    Minimal HTML parser used to extract links from search results.
    """

    def __init__(self) -> None:
        super().__init__()

        self.results: list[dict[str, str]] = []
        self._current_link: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "a":
            return

        attributes = dict(attrs)

        href = attributes.get("href")

        if href:
            self._current_link = href
            self._current_text = []

    def handle_data(
        self,
        data: str,
    ) -> None:
        if self._current_link is not None:
            self._current_text.append(data)

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        if tag != "a":
            return

        if self._current_link is None:
            return

        text = " ".join(
            part.strip()
            for part in self._current_text
            if part.strip()
        ).strip()

        if text:
            self.results.append(
                {
                    "title": text,
                    "url": self._current_link,
                }
            )

        self._current_link = None
        self._current_text = []


def web_search(
    query: str,
    max_results: int = 10,
) -> list[dict[str, str]]:
    """
    Search the public web and return discovered links.

    This implementation intentionally uses Python's standard library
    so the tool does not require an additional search package.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "Search query cannot be empty."
        )

    if max_results < 1:
        raise ValueError(
            "max_results must be >= 1."
        )

    if max_results > 50:
        raise ValueError(
            "max_results must be <= 50."
        )

    search_url = (
        DEFAULT_SEARCH_ENGINE
        + quote_plus(query)
    )

    request = Request(
        search_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/154.0 Safari/537.36"
            )
        },
    )

    try:
        with urlopen(
            request,
            timeout=20,
        ) as response:
            html = response.read().decode(
                "utf-8",
                errors="replace",
            )
    except Exception as exc:
        raise RuntimeError(
            f"Web search failed: {exc}"
        ) from exc

    parser = SearchResultParser()
    parser.feed(html)

    results: list[dict[str, str]] = []

    seen_urls: set[str] = set()

    for result in parser.results:
        url = result["url"]

        if not url.startswith("http"):
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        results.append(result)

        if len(results) >= max_results:
            break

    return results