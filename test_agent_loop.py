from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from agent.loop import AgentLoop, AgentLoopConfig
from config.settings import load_settings
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.ollama_client import OllamaClient
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

# Ollama
model1 = OllamaClient(
    model="beam:cloud",
    base_url=settings.ollama_host,
    timeout=600,
    allow_cloud_models=True,
)
model1.ensure_model_available()

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
Create a retro poem folder and write the complete poem into it 'Twinkle Twinkle Little Star'
"""
task1 = 'hello! what is your knowledge cutoff date.'

original_plan = agent.planner.plan

'''
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
'''
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