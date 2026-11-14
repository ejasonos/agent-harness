from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from agent.loop import AgentLoop, AgentLoopConfig
from config.settings import load_settings
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.factory import create_model_client
from workspace.manager import WorkspaceManager

settings = load_settings()

# Workspace
workspace = WorkspaceManager(
	".",
	allow_delete=settings.allow_delete,
	allow_write=True,
	allow_git_write=settings.allow_git_write,
	allow_shell=settings.allow_shell,
	allow_network=settings.allow_network,
)

# MCP
mcp_client = MCPClient(settings)
registry = ToolRegistry(mcp_client)

model1 = create_model_client(settings, timeout=600)

# Agent
agent = AgentLoop(
    model=model1,
    mcp_client=mcp_client,
    registry=registry,
    workspace=workspace,
    config=AgentLoopConfig(
        max_iterations=settings.max_iterations,
        max_tool_calls=settings.max_tool_calls,
        max_context_messages=settings.max_context_messages,
        max_context_tokens=settings.max_context_tokens,
        model_input_cost_per_million_tokens=(
            settings.model_input_cost_per_million_tokens
        ),
        model_output_cost_per_million_tokens=(
            settings.model_output_cost_per_million_tokens
        ),
        enable_subagents=settings.enable_subagents,
        max_subagents_per_run=settings.max_subagents_per_run,
        max_subagent_iterations=settings.max_subagent_iterations,
        max_subagent_tool_calls=settings.max_subagent_tool_calls,
        max_subagent_task_chars=settings.max_subagent_task_chars,
    ),
)

task = """
Create a retro poem folder and write the complete poem into it 'Twinkle Twinkle Little Star'.
Then tell me the latest about Cristiano Ronaldo
"""

original_plan = agent.planner.plan

state = agent.run(task)

print("\n" + "=" * 30)
print("AGENT RESULT")
print("=" * 30)

print(f"Completed: {state.completed}")
print(f"Iterations: {state.iteration}")
print(f"Tool executions: {len(state.tool_executions)}")
print(f"Tool failure counts: {state.tool_failure_counts}")
print(f"Model calls: {state.model_call_count}")
print(f"Model tokens: {state.model_usage}")
print(f"Estimated model cost (USD): {state.estimated_model_cost_usd}")

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