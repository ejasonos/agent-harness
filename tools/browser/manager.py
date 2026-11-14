from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

MAX_CAPTURED_BROWSER_EVENTS = 1000


@dataclass
class BrowserState:
    console_messages: list[dict[str, Any]] = field(
        default_factory=list
    )

    requests: list[dict[str, Any]] = field(
        default_factory=list
    )

    responses: list[dict[str, Any]] = field(
        default_factory=list
    )

    failed_requests: list[dict[str, Any]] = field(
        default_factory=list
    )


class BrowserManager:
    """
    Owns the Playwright browser session used by the agent.

    One persistent browser context is maintained so that navigation,
    authentication state, cookies, local storage, and application
    state survive across individual tool calls.
    """

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

        self.state = BrowserState()

    # =========================================================
    # LIFECYCLE
    # =========================================================

    async def launch(
        self,
        *,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 900,
    ) -> dict[str, Any]:

        if self._page is not None and not self._page.is_closed():
            return await self.status()

        if self._playwright is not None:
            await self.close()

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=headless,
            )
            self._context = await self._browser.new_context(
                viewport={
                    "width": viewport_width,
                    "height": viewport_height,
                },
            )
            self._context.on("page", self._handle_new_page)
            self._page = await self._context.new_page()
            return await self.status()
        except Exception:
            await self.close()
            raise

    async def close(self) -> dict[str, Any]:
        try:
            if self._browser is not None:
                await self._browser.close()
        finally:
            try:
                if self._playwright is not None:
                    await self._playwright.stop()
            finally:
                self._playwright = None
                self._browser = None
                self._context = None
                self._page = None
                self.state = BrowserState()

        return {
            "closed": True,
        }

    async def status(self) -> dict[str, Any]:

        try:
            page = self.page()
        except RuntimeError:
            return {
                "running": False,
            }

        return {
            "running": True,
            "url": page.url,
            "title": await page.title(),
            "viewport": page.viewport_size,
        }

    # =========================================================
    # INTERNAL ACCESS
    # =========================================================

    def page(self) -> Page:

        if self._page is not None and not self._page.is_closed():
            return self._page

        if self._context is not None:
            open_pages = [
                page
                for page in self._context.pages
                if not page.is_closed()
            ]
            if open_pages:
                self._page = open_pages[-1]
                return self._page

        self._page = None
        if self._context is None:
            raise RuntimeError(
                "Browser is not running. "
                "Call browser_launch first."
            )

        raise RuntimeError(
            "No browser page is open. Call browser_launch to restart the session."
        )

    # =========================================================
    # EVENT LISTENERS
    # =========================================================

    def _install_listeners(
        self,
        page: Page,
    ) -> None:

        page.on(
            "console",
            self._handle_console,
        )

        page.on(
            "request",
            self._handle_request,
        )

        page.on(
            "response",
            self._handle_response,
        )
        page.on(
            "requestfailed",
            self._handle_request_failed,
        )

    def _handle_new_page(
        self,
        page: Page,
    ) -> None:
        self._page = page
        self._install_listeners(page)

    def _append_event(
        self,
        events: list[dict[str, Any]],
        event: dict[str, Any],
    ) -> None:
        events.append(event)
        if len(events) > MAX_CAPTURED_BROWSER_EVENTS:
            del events[:-MAX_CAPTURED_BROWSER_EVENTS]

    def _handle_console(
        self,
        message,
    ) -> None:

        self._append_event(
            self.state.console_messages,
            {
                "type": message.type,
                "text": message.text,
                "location": message.location,
            },
        )

    def _handle_request(
        self,
        request,
    ) -> None:

        self._append_event(
            self.state.requests,
            {
                "method": request.method,
                "url": request.url,
                "resource_type": request.resource_type,
            },
        )

    def _handle_response(
        self,
        response,
    ) -> None:

        self._append_event(
            self.state.responses,
            {
                "status": response.status,
                "status_text": response.status_text,
                "url": response.url,
                "request_method": response.request.method,
            },
        )

    def _handle_request_failed(
        self,
        request,
    ) -> None:
        self._append_event(
            self.state.failed_requests,
            {
                "method": request.method,
                "url": request.url,
                "resource_type": request.resource_type,
                "failure": request.failure,
            },
        )

    # =========================================================
    # STATE
    # =========================================================

    def clear_console(self) -> None:
        self.state.console_messages.clear()

    def clear_network(self) -> None:
        self.state.requests.clear()
        self.state.responses.clear()
        self.state.failed_requests.clear()


browser_manager = BrowserManager()