from agent.state import AgentState, ToolExecution
from model.messages import Message


state = AgentState(
    task="Calculate 25 + 17"
)

state.add_message(
    Message(
        role="user",
        content="Calculate 25 + 17",
    )
)

state.next_iteration()

state.add_tool_execution(
    ToolExecution(
        tool_name="addition_tool",
        arguments={
            "a": 25,
            "b": 17,
        },
        result=42,
        success=True,
        duration_ms=12.5,
    )
)

state.finish("The answer is 42.")

print("Task:", state.task)
print("Iteration:", state.iteration)
print("Completed:", state.completed)
print("Final answer:", state.final_answer)

print("\nTool executions:")

for execution in state.tool_executions:
    print(
        execution.tool_name,
        execution.arguments,
        "=>",
        execution.result,
    )