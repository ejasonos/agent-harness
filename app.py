from __future__ import annotations

from pathlib import Path
from typing import Any

from webmcp_public import mcp


_mcp_app = mcp.http_app(stateless_http=True)
_webmcp_page = Path(__file__).with_name("webmcp.html").read_bytes()


class WebMCPApplication:
    def __init__(self, mcp_app: Any, page: bytes) -> None:
        self.mcp_app = mcp_app
        self.page = page

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            method = scope["method"].upper()
            path = scope.get("path", "/").rstrip("/") or "/"

            if method in {"GET", "HEAD"} and path == "/":
                await self._respond(
                    send,
                    307,
                    b"",
                    [
                        (b"location", b"/mcp"),
                        (b"cache-control", b"no-store"),
                    ],
                )
                return

            if method in {"GET", "HEAD"} and path == "/mcp":
                await self._respond(
                    send,
                    200,
                    self.page,
                    [
                        (b"content-type", b"text/html; charset=utf-8"),
                        (b"cache-control", b"no-store"),
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    ],
                    head=method == "HEAD",
                )
                return

        await self.mcp_app(scope, receive, send)

    @staticmethod
    async def _respond(
        send,
        status: int,
        body: bytes,
        headers: list[tuple[bytes, bytes]],
        *,
        head: bool = False,
    ) -> None:
        response_headers = headers + [
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": response_headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"" if head else body,
            }
        )


app = WebMCPApplication(_mcp_app, _webmcp_page)
