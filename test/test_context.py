from agent.context import ContextManager
from agent.state import AgentState


state = AgentState(
    task="Calculate 25 + 17"
)

context = ContextManager(
    max_messages=5
)

context.add_user_message(
    state,
    "Calculate 25 + 17"
)

context.add_assistant_message(
    state,
    "I will calculate that."
)

context.add_tool_message(
    state,
    "42",
    tool_name="addition_tool"
)

messages = context.build_messages(state)

print("Messages in context:")

for message in messages:
    print(
        f"[{message.role}] {message.content}"
    )

print("\nContext message count:", len(messages))