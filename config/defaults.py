"""
Default configuration for the local agent runtime.
"""

DEFAULT_MODEL = "gemma4:e4b"

DEFAULT_MCP_URL = "http://127.0.0.1:8082/mcp"

# Agent loop limits
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_MAX_TOOL_CALLS = 100
DEFAULT_MAX_CONTEXT_MESSAGES = 50

# Execution limits
DEFAULT_COMMAND_TIMEOUT = 120
DEFAULT_TOOL_TIMEOUT = 120

# Browser defaults
DEFAULT_BROWSER_WIDTH = 1440
DEFAULT_BROWSER_HEIGHT = 900

# Common responsive viewports
VIEWPORTS = {
    "desktop": {
        "width": 1440,
        "height": 900,
    },
    "laptop": {
        "width": 1280,
        "height": 800,
    },
    "tablet": {
        "width": 768,
        "height": 1024,
    },
    "mobile": {
        "width": 390,
        "height": 844,
    },
    "small_mobile": {
        "width": 320,
        "height": 568,
    },
}

# Output limits
MAX_FILE_READ_BYTES = 1_000_000
MAX_TOOL_OUTPUT_CHARS = 50_000
MAX_TERMINAL_OUTPUT_CHARS = 30_000
MAX_WEB_CONTENT_CHARS = 40_000

# Workspace security
ALLOW_DELETE = True
ALLOW_GIT_WRITE = False
ALLOW_NETWORK = True
ALLOW_SHELL = True

# Logging
LOG_LEVEL = "INFO"