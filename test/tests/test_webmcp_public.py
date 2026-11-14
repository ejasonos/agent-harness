import asyncio
import json

import pytest
from fastmcp import Client
from starlette.testclient import TestClient

from app import app
from webmcp_public import (
    mcp,
    get_project_overview,
    read_project_doc,
    search_project_docs,
)


def test_public_documentation_tools_are_read_only_and_allowlisted():
    overview = get_project_overview()
    workspace_safety = read_project_doc("workspace-safety")

    assert overview["title"] == "Local Agent"
    assert "model-directed AI agent" in overview["text"]
    assert workspace_safety["title"] == "Workspace Safety"
    assert "workspace" in workspace_safety["text"].lower()

    with pytest.raises(ValueError, match="section must be one of"):
        read_project_doc("../.env")


def test_public_documentation_search_is_bounded():
    results = search_project_docs("shell", limit=2)

    assert 0 < len(results) <= 2
    assert all(result["section"] in {
        "available-tool-groups",
        "workspace-safety",
    } for result in results)
    assert all(len(result["excerpt"]) <= 500 for result in results)
    assert all("```" not in result["excerpt"] for result in results)

    with pytest.raises(ValueError, match="160 characters"):
        search_project_docs("x" * 161)
    with pytest.raises(ValueError, match="between 1 and 10"):
        search_project_docs("shell", limit=11)


def test_public_mcp_endpoint_lists_only_documentation_tools():
    async def list_tool_names():
        async with Client(mcp) as client:
            return {tool.name for tool in await client.list_tools()}

    assert asyncio.run(list_tool_names()) == {
        "get_project_overview",
        "search_project_docs",
        "read_project_doc",
    }


def test_vercel_app_serves_page_and_read_only_mcp_tools():
    with TestClient(app) as client:
        page = client.get("/mcp")
        assert page.status_code == 200
        assert "Local Agent WebMCP" in page.text

        root = client.get("/", follow_redirects=False)
        assert root.status_code == 307
        assert root.headers["location"] == "/mcp"

        headers = {"Accept": "application/json, text/event-stream"}
        initialize = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "webmcp-test", "version": "1"},
                },
            },
        )
        assert initialize.status_code == 200

        initialized = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert initialized.status_code == 202

        response = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_project_overview",
                    "arguments": {},
                },
            },
        )
        assert response.status_code == 200
        message = next(
            json.loads(line.removeprefix("data:").strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
        )
        assert message["result"]["structuredContent"]["title"] == "Local Agent"
