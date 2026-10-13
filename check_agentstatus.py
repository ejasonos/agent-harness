from __future__ import annotations

import asyncio
import os

from fastmcp import Client


SERVER_URL = os.getenv(
    "AGENTSTATUS_MCP_URL",
    "https://agent-harness.fastmcp.app/mcp",
)


async def main() -> None:
    async with Client(SERVER_URL, auth="oauth") as client:
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools}
        print(f"Connected to {SERVER_URL}")
        print(f"Discovered {len(tools)} tools")

        if "addition" not in tool_names:
            raise RuntimeError("The server does not expose the 'addition' tool.")

        result = await client.call_tool("addition", {"a": 2, "b": 3})
        if result.is_error:
            raise RuntimeError(f"addition tool failed: {result.content}")

        print(f"addition(2, 3) returned: {result.data}")
        print("Authenticated MCP tool call succeeded.")


if __name__ == "__main__":
    asyncio.run(main())