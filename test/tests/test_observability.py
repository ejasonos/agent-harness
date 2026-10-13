import io
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

from mcp_client.client import MCPClient
from utils.logging import log_context, log_event


def test_log_event_carries_run_context_and_redacts_credentials(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    logger = logging.getLogger("local_agent")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)

    try:
        with log_context(run_id="run-123", model_purpose="planner"):
            log_event(
                "test.event",
                credentials={
                    "api_key": "must-not-appear",
                    "safe": "visible",
                },
            )
    finally:
        logger.removeHandler(handler)

    record = json.loads(stream.getvalue())
    assert record["run_id"] == "run-123"
    assert record["model_purpose"] == "planner"
    assert record["credentials"]["api_key"] == "[REDACTED]"
    assert record["credentials"]["safe"] == "visible"


def test_mcp_http_trace_records_request_and_response(monkeypatch):
    events = []
    monkeypatch.setattr(
        "mcp_client.client.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    client = MCPClient(
        SimpleNamespace(
            mcp_url="http://localhost/mcp",
            tool_timeout=15,
        )
    )
    client.initialized = True
    response = Mock(
        status_code=200,
        headers={"content-type": "application/json"},
        text='{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}',
    )
    response.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": []},
    }
    client.http.post = Mock(return_value=response)

    result = client.request("tools/list", {})

    assert result == {"tools": []}
    event_names = [event for event, _fields in events]
    assert "mcp.request.sent" in event_names
    assert "mcp.request.payload" in event_names
    assert "mcp.response.received" in event_names
    assert "mcp.response.payload" in event_names
    request = next(
        fields
        for event, fields in events
        if event == "mcp.request.sent"
    )
    assert request["method"] == "tools/list"
    assert request["request_id"] == 1
