from __future__ import annotations

from urllib.request import Request, urlopen


DEFAULT_TIMEOUT = 30
MAX_CONTENT_SIZE = 5 * 1024 * 1024


def open_url(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """
    Retrieve a public URL and return its response metadata and content.
    """

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

    if timeout < 1:
        raise ValueError(
            "timeout must be >= 1 second."
        )

    if timeout > 120:
        raise ValueError(
            "timeout must be <= 120 seconds."
        )

    request = Request(
        url,
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
            timeout=timeout,
        ) as response:

            content_type = response.headers.get(
                "Content-Type",
                "",
            )

            final_url = response.geturl()

            data = response.read(
                MAX_CONTENT_SIZE + 1
            )

            truncated = (
                len(data) > MAX_CONTENT_SIZE
            )

            if truncated:
                data = data[
                    :MAX_CONTENT_SIZE
                ]

            charset = response.headers.get_content_charset()

            if charset is None:
                charset = "utf-8"

            content = data.decode(
                charset,
                errors="replace",
            )

            return {
                "url": url,
                "final_url": final_url,
                "status_code": response.status,
                "content_type": content_type,
                "content_length": len(data),
                "truncated": truncated,
                "content": content,
            }

    except Exception as exc:
        raise RuntimeError(
            f"Failed to open URL: {exc}"
        ) from exc