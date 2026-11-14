import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import fastmcp_tools_http


def test_missing_agentstatus_endpoint_does_not_break_tool_call(monkeypatch):
    monkeypatch.setattr(fastmcp_tools_http, "AGENTSTATUS_URL", None)
    urlopen = Mock(side_effect=AssertionError("telemetry should be disabled"))
    monkeypatch.setattr(fastmcp_tools_http, "urlopen", urlopen)
    context = SimpleNamespace(
        message=SimpleNamespace(name="directory_tree")
    )

    async def call_next(_context):
        return "workspace tree"

    result = asyncio.run(
        fastmcp_tools_http.AgentStatusMiddleware().on_call_tool(
            context,
            call_next,
        )
    )

    assert result == "workspace tree"
    urlopen.assert_not_called()