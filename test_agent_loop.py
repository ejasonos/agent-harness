from config.settings import load_settings

from agent.loop import AgentLoop
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.ollama_client import OllamaClient
from model.nvidia_client import NVIDIAClient

from workspace.manager import WorkspaceManager

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


settings = load_settings()
# Workspace
workspace = WorkspaceManager(
    ".",
    allow_delete=settings.allow_delete,
)

# MCP
mcp_client = MCPClient(settings)
registry = ToolRegistry(mcp_client)

# Ollama
model1 = OllamaClient(
    model=settings.model,
    base_url=settings.ollama_host,
    timeout=600,
)
model2 = NVIDIAClient(
    model=settings.nvidia_model,
    api_key=settings.nvidia_api_key,
    base_url=settings.nvidia_endpoint,
    timeout=600,
)

# Agent
agent = AgentLoop(
    model=model2,
    mcp_client=mcp_client,
    registry=registry,
    workspace=workspace
)

task = """
Task: create a poem folder and create three different poem files in it having loving poems with two stanzas each. The stanzas should be about love and magic.

You are an AI Agent that performs tasks one by one.

You have access to tools.

You cannot complete the task in one go, or one iteration. You must break the task down into smaller steps and complete them one by one.

The task should be completed in this order:
1. Create a folder named poem_folder
2. Create three different poem.txt files in the poem_folder
3. Write unique poems in each of the files, two stanzas following the theme provided in the task description.

Don't stop till you are done with the task
"""

state = agent.run(task)

print("\n" + "=" * 60)
print("AGENT RESULT")
print("=" * 60)

print(f"Completed: {state.completed}")
print(f"Iterations: {state.iteration}")
print(f"Tool executions: {len(state.tool_executions)}")

print("\nTool executions:")

for execution in state.tool_executions:
    print(f"\nTool: {execution.tool_name}")
    print(f"Reason: {execution.reason_for_tool or 'Not provided'}")
    print(f"Arguments: {execution.arguments}")
    print(f"Success: {execution.success}")
    print(f"Result: {execution.result}")

print("\nFinal answer:")
print(state.final_answer)

if state.error:
    print("\nError:")
    print(state.error)