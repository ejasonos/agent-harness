from config.settings import load_settings

from agent.loop import AgentLoop, AgentLoopConfig
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
    workspace=workspace,
    config=AgentLoopConfig(
        max_iterations=300,
        max_tool_calls=300,
    ),
)

task = """
You are an AI Agent that performs tasks one by one.

You have access to tools.

You cannot complete the task in one go, or one iteration. You must break the task down into smaller steps and complete them one by one.

Make special use of web_search tool for current information and data.

The task should be completed in this order:
1. Create a folder named poem_folder
2. Create three different poem.txt files in the poem_folder
3. For each file, create a 500 words long poem with two stanzas following the theme provided in the task description.
4. After completing the task, provide a final answer summarizing what you have done.
5. Search about Ronaldo and Portugal reef that is ongoing.
6. Name all the current forbes richest people and their net worth.

Give an analysis of the whole task from one to six, give the analysis of each task separately.
Don't stop till you are done with the task
"""

original_plan = agent.planner.plan


def plan_with_live_output(state, tools=None):
    plan = original_plan(state, tools=tools)
    response = plan.response

    if response.content:
        print(f"\n[MODEL]\n{response.content}", flush=True)

    for tool_call in response.tool_calls:
        print(
            f"\n[MODEL -> TOOL] {tool_call.tool_name}: "
            f"{tool_call.arguments}",
            flush=True,
        )

    return plan


agent.planner.plan = plan_with_live_output

original_execute_tool = agent._execute_tool


def execute_tool_with_live_output(state, tool_call):
    print(f"\n[TOOL START] {tool_call.tool_name}", flush=True)
    execution = original_execute_tool(state, tool_call)

    status = "success" if execution.success else "failed"
    print(f"[TOOL {status.upper()}] {tool_call.tool_name}", flush=True)
    if execution.error:
        print(f"[TOOL ERROR]\n{execution.error}", flush=True)
    elif execution.result is not None:
        print(f"[TOOL RESULT]\n{execution.result}", flush=True)

    return execution


agent._execute_tool = execute_tool_with_live_output

state = agent.run(task)

print("\n" + "=" * 60)
print("AGENT RESULT")
print("=" * 60)

print(f"Completed: {state.completed}")
print(f"Iterations: {state.iteration}")
print(f"Tool executions: {len(state.tool_executions)}")
print(f"Tool failure counts: {state.tool_failure_counts}")

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