from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)


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

    def launch(
        self,
        *,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 900,
    ) -> dict[str, Any]:

        if self._page is not None:
            return self.status()

        self._playwright = sync_playwright().start()

        self._browser = self._playwright.chromium.launch(
            headless=headless,
        )

        self._context = self._browser.new_context(
            viewport={
                "width": viewport_width,
                "height": viewport_height,
            },
        )

        self._page = self._context.new_page()

        self._install_listeners(
            self._page
        )

        return self.status()

    def close(self) -> dict[str, Any]:

        if self._browser is not None:
            self._browser.close()

        if self._playwright is not None:
            self._playwright.stop()

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

        self.state = BrowserState()

        return {
            "closed": True,
        }

    def status(self) -> dict[str, Any]:

        if self._page is None:
            return {
                "running": False,
            }

        return {
            "running": True,
            "url": self._page.url,
            "title": self._page.title(),
            "viewport": self._page.viewport_size,
        }

    # =========================================================
    # INTERNAL ACCESS
    # =========================================================

    def page(self) -> Page:

        if self._page is None:
            raise RuntimeError(
                "Browser is not running. "
                "Call browser_launch first."
            )

        return self._page

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

    def _handle_console(
        self,
        message,
    ) -> None:

        self.state.console_messages.append(
            {
                "type": message.type,
                "text": message.text,
                "location": message.location,
            }
        )

    def _handle_request(
        self,
        request,
    ) -> None:

        self.state.requests.append(
            {
                "method": request.method,
                "url": request.url,
                "resource_type": request.resource_type,
            }
        )

    def _handle_response(
        self,
        response,
    ) -> None:

        self.state.responses.append(
            {
                "status": response.status,
                "status_text": response.status_text,
                "url": response.url,
                "request_method": response.request.method,
            }
        )

    # =========================================================
    # STATE
    # =========================================================

    def clear_console(self) -> None:
        self.state.console_messages.clear()

    def clear_network(self) -> None:
        self.state.requests.clear()
        self.state.responses.clear()


browser_manager = BrowserManager()