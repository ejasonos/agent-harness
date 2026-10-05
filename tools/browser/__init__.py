from .lifecycle import (
    browser_launch,
    browser_close,
    browser_status,
)

from .navigation import (
    browser_goto,
    browser_back,
    browser_forward,
    browser_reload,
    browser_current_page,
)

from .interaction import (
    browser_click,
    browser_double_click,
    browser_fill,
    browser_type,
    browser_press,
    browser_select,
    browser_check,
    browser_uncheck,
    browser_hover,
)

from .inspection import (
    browser_page_text,
    browser_element_text,
    browser_element_attribute,
    browser_element_exists,
    browser_page_html,
)

from .console import (
    browser_get_console,
    browser_clear_console,
)

from .network import (
    browser_get_requests,
    browser_get_responses,
    browser_clear_network,
)

from .viewport import (
    browser_screenshot,
    browser_set_viewport,
    browser_scroll,
)

__all__ = [
    "browser_launch",
    "browser_close",
    "browser_status",
    "browser_goto",
    "browser_back",
    "browser_forward",
    "browser_reload",
    "browser_current_page",
    "browser_click",
    "browser_double_click",
    "browser_fill",
    "browser_type",
    "browser_press",
    "browser_select",
    "browser_check",
    "browser_uncheck",
    "browser_hover",
    "browser_page_text",
    "browser_element_text",
    "browser_element_attribute",
    "browser_element_exists",
    "browser_page_html",
    "browser_get_console",
    "browser_clear_console",
    "browser_get_requests",
    "browser_get_responses",
    "browser_clear_network",
    "browser_screenshot",
    "browser_set_viewport",
    "browser_scroll",
]