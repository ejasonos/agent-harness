"""
FastMCP HTTP client.

Responsibilities:

- Establish MCP sessions
- Initialize the MCP connection
- Discover tools
- Execute tools
- Handle MCP errors
- Handle JSON and SSE responses
"""

from __future__ import annotations

import json
import threading
from typing import Any

import requests

from config.settings import AgentSettings
from .schemas import MCPToolResult


class MCPError(RuntimeError):
    """Raised when an MCP operation fails."""


class MCPClient:
    """
    HTTP client for a FastMCP server.
    """

    def __init__(
        self,
        settings: AgentSettings,
    ):
        self.settings = settings

        self.url = settings.mcp_url

        self.session_id: str | None = None

        self.initialized = False

        self._request_id = 0

        self._lock = threading.Lock()

        self.http = requests.Session()

        self.http.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
        )

    # ------------------------------------------------------------------
    # Request IDs
    # ------------------------------------------------------------------

    def _next_request_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _post(
        self,
        payload: dict[str, Any],
        timeout: int | None = None,
    ) -> Any:

        headers = {}

        if self.session_id:
            headers["mcp-session-id"] = self.session_id

        timeout = timeout or self.settings.tool_timeout

        try:
            response = self.http.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )

        except requests.RequestException as exc:
            raise MCPError(
                f"Unable to connect to MCP server: {exc}"
            ) from exc

        # FastMCP may establish/update the session through a header.
        session_id = response.headers.get(
            "mcp-session-id"
        )

        if session_id:
            self.session_id = session_id

        if response.status_code >= 400:
            raise MCPError(
                f"MCP HTTP {response.status_code}: "
                f"{response.text[:2000]}"
            )

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: requests.Response,
    ) -> Any:

        text = response.text.strip()

        if not text:
            return {}

        content_type = (
            response.headers.get(
                "content-type",
                "",
            )
            .lower()
        )

        # Normal JSON response.
        if (
            "application/json" in content_type
            or text.startswith("{")
            or text.startswith("[")
        ):
            try:
                return response.json()

            except ValueError:
                pass

        # SSE response.
        if (
            "text/event-stream" in content_type
            or text.startswith("event:")
            or text.startswith("data:")
        ):
            return self._parse_sse(text)

        return {
            "raw": text,
        }

    def _parse_sse(
        self,
        text: str,
    ) -> Any:

        messages: list[Any] = []

        current_data: list[str] = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                if current_data:
                    data = "\n".join(
                        current_data
                    )

                    try:
                        messages.append(
                            json.loads(data)
                        )
                    except json.JSONDecodeError:
                        messages.append(data)

                    current_data = []

                continue

            if line.startswith("data:"):
                current_data.append(
                    line[5:].strip()
                )

        if current_data:
            data = "\n".join(current_data)

            try:
                messages.append(
                    json.loads(data)
                )
            except json.JSONDecodeError:
                messages.append(data)

        if len(messages) == 1:
            return messages[0]

        return messages

    # ------------------------------------------------------------------
    # JSON-RPC
    # ------------------------------------------------------------------

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int | None = None,
    ) -> Any:

        request_id = self._next_request_id()

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }

        if params is not None:
            payload["params"] = params

        result = self._post(
            payload,
            timeout=timeout,
        )

        return self._unwrap_result(
            result
        )

    def notification(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> Any:

        payload = {
            "jsonrpc": "2.0",
            "method": method,
        }

        if params is not None:
            payload["params"] = params

        return self._post(payload)

    def _unwrap_result(
        self,
        response: Any,
    ) -> Any:

        # SSE may return multiple messages.
        if isinstance(response, list):

            for item in reversed(response):

                if isinstance(item, dict):
                    if "error" in item:
                        self._raise_rpc_error(
                            item
                        )

                    if "result" in item:
                        return item["result"]

            return response

        if not isinstance(response, dict):
            return response

        if "error" in response:
            self._raise_rpc_error(
                response
            )

        if "result" in response:
            return response["result"]

        return response

    def _raise_rpc_error(
        self,
        response: dict[str, Any],
    ) -> None:

        error = response.get(
            "error",
            {},
        )

        code = error.get(
            "code",
            "unknown",
        )

        message = error.get(
            "message",
            "Unknown MCP error",
        )

        data = error.get(
            "data"
        )

        detail = (
            f" ({data})"
            if data is not None
            else ""
        )

        raise MCPError(
            f"MCP JSON-RPC error "
            f"{code}: {message}{detail}"
        )

    # ------------------------------------------------------------------
    # MCP lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> dict[str, Any]:

        result = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "local-agent",
                    "version": "1.0.0",
                },
            },
        )

        self.notification(
            "notifications/initialized"
        )

        self.initialized = True

        return result

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def list_tools(
        self,
    ) -> list[dict[str, Any]]:
        
        if not self.initialized:
            self.initialize()

        result = self.request(
            "tools/list",
            {},
        )

        if isinstance(result, dict):
            tools = result.get(
                "tools",
                [],
            )

            if isinstance(tools, list):
                return tools

        return []

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:

        if not self.initialized:
            self.initialize()

        arguments = arguments or {}

        try:

            result = self.request(
                "tools/call",
                {
                    "name": name,
                    "arguments": arguments,
                },
            )

            if not isinstance(result, dict):
                return MCPToolResult(
                    tool_name=name,
                    success=True,
                    content=result,
                    raw=result,
                )

            is_error = bool(
                result.get(
                    "isError",
                    False,
                )
            )

            content = result.get(
                "content"
            )

            structured = result.get(
                "structuredContent"
            )

            return MCPToolResult(
                tool_name=name,
                success=not is_error,
                content=content,
                structured_content=structured,
                error=(
                    self._extract_error_content(
                        content
                    )
                    if is_error
                    else None
                ),
                raw=result,
            )

        except MCPError as exc:

            return MCPToolResult(
                tool_name=name,
                success=False,
                error=str(exc),
            )

    def _extract_error_content(
        self,
        content: Any,
    ) -> str:

        if isinstance(content, str):
            return content

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):

                    text = item.get(
                        "text"
                    )

                    if text:
                        parts.append(
                            str(text)
                        )

            if parts:
                return "\n".join(parts)

        return str(content)