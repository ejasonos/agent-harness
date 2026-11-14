import asyncio

import pytest

from tools.browser.inspection import browser_page_snapshot
from tools.browser.manager import BrowserManager
from tools.browser.waiting import browser_wait


class FakeLocator:
    def __init__(self):
        self.waited_for = None
        self.evaluated_with = None

    async def wait_for(self, *, state, timeout):
        self.waited_for = (state, timeout)

    async def evaluate(self, script):
        self.evaluated_with = script
        return {
            "headings": [{"level": 1, "text": "Account"}],
            "controls": [
                {
                    "role": "button",
                    "name": "Continue",
                    "selector": 'role=button[name="Continue"]',
                }
            ],
            "control_count": 1,
            "controls_truncated": False,
        }


class FakePage:
    def __init__(self, url="about:blank", closed=False):
        self.url = url
        self.closed = closed
        self.load_state = None
        self.locator_instance = FakeLocator()
        self.listeners = {}

    def is_closed(self):
        return self.closed

    def locator(self, selector):
        self.selector = selector
        return self.locator_instance

    async def wait_for_load_state(self, state, *, timeout):
        self.load_state = (state, timeout)

    async def title(self):
        return "Test page"

    def on(self, event, callback):
        self.listeners[event] = callback


def test_browser_wait_waits_for_selector_state(monkeypatch):
    page = FakePage(url="https://example.test")
    monkeypatch.setattr(
        "tools.browser.waiting.browser_manager.page",
        lambda: page,
    )

    result = asyncio.run(
        browser_wait(
            selector="role=button[name='Continue']",
            state="visible",
            timeout=2500,
        )
    )

    assert page.selector == "role=button[name='Continue']"
    assert page.locator_instance.waited_for == ("visible", 2500)
    assert result["success"] is True
    assert result["state"] == "visible"


def test_browser_wait_waits_for_load_state(monkeypatch):
    page = FakePage(url="https://example.test")
    monkeypatch.setattr(
        "tools.browser.waiting.browser_manager.page",
        lambda: page,
    )

    asyncio.run(browser_wait(state="networkidle", timeout=3500))

    assert page.load_state == ("networkidle", 3500)


def test_browser_wait_rejects_mismatched_selector_state(monkeypatch):
    page = FakePage()
    monkeypatch.setattr(
        "tools.browser.waiting.browser_manager.page",
        lambda: page,
    )

    with pytest.raises(ValueError, match="With a selector"):
        asyncio.run(browser_wait(selector="main", state="load"))


def test_browser_snapshot_returns_interactive_locator_hints(monkeypatch):
    page = FakePage(url="https://example.test")
    monkeypatch.setattr(
        "tools.browser.inspection.browser_manager.page",
        lambda: page,
    )

    snapshot = asyncio.run(browser_page_snapshot())

    assert snapshot["url"] == "https://example.test"
    assert snapshot["headings"] == [{"level": 1, "text": "Account"}]
    assert snapshot["controls"][0]["selector"] == (
        'role=button[name="Continue"]'
    )
    assert "element.value" not in page.locator_instance.evaluated_with


def test_browser_manager_tracks_latest_page_and_failed_requests():
    manager = BrowserManager()
    first_page = FakePage("https://example.test")
    popup = FakePage("https://example.test/popup")
    first_page.closed = True
    manager._context = type("Context", (), {"pages": [first_page, popup]})()
    manager._page = first_page
    manager._handle_new_page(popup)

    request = type(
        "Request",
        (),
        {
            "method": "GET",
            "url": "https://example.test/api",
            "resource_type": "xhr",
            "failure": "net::ERR_CONNECTION_REFUSED",
        },
    )()
    manager._handle_request_failed(request)

    assert manager.page() is popup
    assert manager.state.failed_requests == [
        {
            "method": "GET",
            "url": "https://example.test/api",
            "resource_type": "xhr",
            "failure": "net::ERR_CONNECTION_REFUSED",
        }
    ]
    assert set(popup.listeners) == {
        "console",
        "request",
        "response",
        "requestfailed",
    }