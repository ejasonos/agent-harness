from agent.context import ContextManager
from agent.planner import Planner
from agent.state import AgentState
from model.ollama_client import OllamaClient


state = AgentState(
    task="What is 25 + 17?"
)

context = ContextManager()

context.add_user_message(
    state,
    state.task,
)

model = OllamaClient()

planner = Planner(
    model=model,
    context=context,
)

plan = planner.plan(state)

print("Model response:")
print(plan.response.content)

print("\nTool calls:")
print(plan.response.tool_calls)

print("\nShould use tools:")
print(plan.should_use_tools)

print("\nShould finish:")
print(plan.should_finish)