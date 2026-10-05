from config.settings import load_settings
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry

settings = load_settings()
client = MCPClient(settings)

print("\n=== INITIALIZING MCP ===")
info = client.initialize()
print(info)

print("\n=== DISCOVERING TOOLS ===")
registry = ToolRegistry(client)
tools = registry.discover()

for tool in tools:
    print(f"\n{tool.name}")
    print(f"  {tool.description}")
    print(f"  {tool.input_schema}")

print("\n=== TOOL NAMES ===")
print([tool.name for tool in registry.all()])