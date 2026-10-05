"""
Runtime configuration for the local agent.

Environment variables can override the defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from .defaults import (
    DEFAULT_MODEL,
    DEFAULT_MCP_URL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_CONTEXT_MESSAGES,
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_TOOL_TIMEOUT,
    DEFAULT_BROWSER_WIDTH,
    DEFAULT_BROWSER_HEIGHT,
    MAX_FILE_READ_BYTES,
    MAX_TOOL_OUTPUT_CHARS,
    MAX_TERMINAL_OUTPUT_CHARS,
    MAX_WEB_CONTENT_CHARS,
    ALLOW_DELETE,
    ALLOW_GIT_WRITE,
    ALLOW_NETWORK,
    ALLOW_SHELL,
    LOG_LEVEL,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class AgentSettings:
    """
    Global configuration for an agent session.
    """

    # Model
    model: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_MODEL",
            DEFAULT_MODEL,
        )
    )

    ollama_host: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_HOST",
            "http://127.0.0.1:11434",
        )
    )

    nvidia_api_key: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_API_KEY",
            ""
        )
    )

    nvidia_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_ENDPOINT",
            "https://integrate.api.nvidia.com/v1"
        )
    )

    nvidia_model: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_MODEL",
            "meta/llama-3.2-11b-vision-instruct",
        )
    )

    # MCP
    mcp_url: str = field(
        default_factory=lambda: os.getenv(
            "MCP_URL",
            DEFAULT_MCP_URL,
        )
    )

    # Agent loop
    max_iterations: int = field(
        default_factory=lambda: _env_int(
            "MAX_ITERATIONS",
            DEFAULT_MAX_ITERATIONS,
        )
    )

    max_tool_calls: int = field(
        default_factory=lambda: _env_int(
            "MAX_TOOL_CALLS",
            DEFAULT_MAX_TOOL_CALLS,
        )
    )

    max_context_messages: int = field(
        default_factory=lambda: _env_int(
            "MAX_CONTEXT_MESSAGES",
            DEFAULT_MAX_CONTEXT_MESSAGES,
        )
    )

    # Execution
    command_timeout: int = field(
        default_factory=lambda: _env_int(
            "COMMAND_TIMEOUT",
            DEFAULT_COMMAND_TIMEOUT,
        )
    )

    tool_timeout: int = field(
        default_factory=lambda: _env_int(
            "TOOL_TIMEOUT",
            DEFAULT_TOOL_TIMEOUT,
        )
    )

    # Browser
    browser_width: int = field(
        default_factory=lambda: _env_int(
            "BROWSER_WIDTH",
            DEFAULT_BROWSER_WIDTH,
        )
    )

    browser_height: int = field(
        default_factory=lambda: _env_int(
            "BROWSER_HEIGHT",
            DEFAULT_BROWSER_HEIGHT,
        )
    )

    # Output limits
    max_file_read_bytes: int = MAX_FILE_READ_BYTES
    max_tool_output_chars: int = MAX_TOOL_OUTPUT_CHARS
    max_terminal_output_chars: int = MAX_TERMINAL_OUTPUT_CHARS
    max_web_content_chars: int = MAX_WEB_CONTENT_CHARS

    # Permissions
    allow_delete: bool = field(
        default_factory=lambda: _env_bool(
            "ALLOW_DELETE",
            ALLOW_DELETE,
        )
    )

    allow_git_write: bool = field(
        default_factory=lambda: _env_bool(
            "ALLOW_GIT_WRITE",
            ALLOW_GIT_WRITE,
        )
    )

    allow_network: bool = field(
        default_factory=lambda: _env_bool(
            "ALLOW_NETWORK",
            ALLOW_NETWORK,
        )
    )

    allow_shell: bool = field(
        default_factory=lambda: _env_bool(
            "ALLOW_SHELL",
            ALLOW_SHELL,
        )
    )

    # Logging
    log_level: str = field(
        default_factory=lambda: os.getenv(
            "LOG_LEVEL",
            LOG_LEVEL,
        )
    )

    # Workspace
    workspace: Path | None = None

    def set_workspace(self, path: str | Path) -> Path:
        """
        Set and validate the active workspace.
        """

        workspace = Path(path).expanduser().resolve()

        if not workspace.exists():
            raise FileNotFoundError(
                f"Workspace does not exist: {workspace}"
            )

        if not workspace.is_dir():
            raise NotADirectoryError(
                f"Workspace is not a directory: {workspace}"
            )

        self.workspace = workspace

        return workspace

    def require_workspace(self) -> Path:
        """
        Return the active workspace or raise an error.
        """

        if self.workspace is None:
            raise RuntimeError(
                "No workspace configured. "
                "Start the agent with a workspace path."
            )

        return self.workspace


def load_settings(
    workspace: str | Path | None = None,
) -> AgentSettings:
    """
    Create the runtime settings.

    Example:

        settings = load_settings(
            r"C:\\Projects\\my-node-app"
        )
    """

    settings = AgentSettings()

    if workspace is not None:
        settings.set_workspace(workspace)

    return settings