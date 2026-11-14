import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import load_settings
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry


settings = load_settings()

client = MCPClient(settings)

registry = ToolRegistry(client)

tools = registry.discover()
tool_names = {tool.name for tool in tools}
required_tools = {
    "read_files",
    "browser_launch",
    "browser_wait",
    "browser_page_snapshot",
    "browser_get_failed_requests",
    "web_search",
}
missing_tools = required_tools - tool_names
if missing_tools:
    raise RuntimeError(
        "MCP server is missing expected tools: "
        + ", ".join(sorted(missing_tools))
    )

print("Discovered tools:")
print()

for tool in tools:
    print(f"Name: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Schema: {tool.input_schema}")
    print()

print("OpenAI-compatible tool definitions:")
print()

for tool in registry.to_openai_tools():
    print(tool)