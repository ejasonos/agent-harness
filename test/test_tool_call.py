from config.settings import load_settings
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.ollama_client import OllamaClient
from model.messages import Message


settings = load_settings()

# MCP
mcp_client = MCPClient(settings)
registry = ToolRegistry(mcp_client)

tools = registry.discover()
ollama_tools = registry.to_ollama_tools()

print("Available tools:")
for tool in tools:
    print(f"- {tool.name}")

# Model
model = OllamaClient(
    model=settings.model,
    base_url=settings.ollama_host,
    timeout=120
)

# Ask the model to use a tool
messages = [
    Message(
        role="user",
        content="Use the addition_tool to calculate 25 + 17."
    )
]

response = model.chat(
    messages=messages,
    tools=ollama_tools,
)

print("\nModel response:")
print(response.content)

print("\nTool calls:")
for call in response.tool_calls:
    print(f"Tool: {call.tool_name}")
    print(f"Arguments: {call.arguments}")
    print(f"Call ID: {call.call_id}")

    result = mcp_client.call_tool(
        call.tool_name,
        call.arguments,
    )

    print("\nTool result:")
    print(f"Success: {result.success}")
    print(f"Content: {result.as_text()}")