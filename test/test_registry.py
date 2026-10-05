from config.settings import load_settings
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry


settings = load_settings()

client = MCPClient(settings)

registry = ToolRegistry(client)

tools = registry.discover()

print("Discovered tools:")
print()

for tool in tools:
    print(f"Name: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Schema: {tool.input_schema}")
    print()

print("Ollama tool definitions:")
print()

for tool in registry.to_ollama_tools():
    print(tool)