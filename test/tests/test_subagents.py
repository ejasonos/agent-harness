from types import SimpleNamespace

from agent.loop import AgentLoop, AgentLoopConfig
from agent.state import AgentState, ToolExecution
from agent.subagents import ReadOnlyToolRegistry, SubagentRunner
from mcp_client.schemas import MCPTool
from model.messages import ModelResponse, ToolCall


class FakeRegistry:
    def __init__(self):
        self.tools = {
            name: MCPTool(name=name)
            for name in (
                "read_file",
                "git_status",
                "write_file",
                "execute_command",
                "delegate_task",
            )
        }

    def all(self):
        return list(self.tools.values())

    def discover(self):
        return self.all()

    def get(self, name):
        return self.tools.get(name)

    def to_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {"name": name, "parameters": {}},
            }
            for name in self.tools
        ]


def test_child_registry_exposes_only_read_only_inspection_tools():
    registry = ReadOnlyToolRegistry(FakeRegistry())

    assert {tool.name for tool in registry.discover()} == {
        "read_file",
        "git_status",
    }
    assert registry.get("write_file") is None
    assert registry.get("execute_command") is None
    assert registry.get("delegate_task") is None
    assert {
        schema["function"]["name"]
        for schema in registry.to_openai_tools()
    } == {"read_file", "git_status"}


def test_subagent_child_has_stricter_limits_and_cannot_recurse(monkeypatch):
    captured = {}

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, task):
            captured["task"] = task
            return SimpleNamespace(completed=True, error=None)

    monkeypatch.setattr("agent.loop.AgentLoop", FakeAgentLoop)
    config = AgentLoopConfig(
        max_context_tokens=4000,
        max_subagent_iterations=5,
        max_subagent_tool_calls=7,
    )
    runner = SubagentRunner(
        model=object(),
        mcp_client=object(),
        registry=FakeRegistry(),
        workspace=object(),
        config=config,
    )

    result = runner.run("summarize project", "inspect its README")

    child_config = captured["config"]
    assert result.completed
    assert child_config.enable_subagents is False
    assert child_config.max_iterations == 5
    assert child_config.max_tool_calls == 7
    assert child_config.max_context_tokens == 4000
    assert "Do not modify files" in captured["task"]
    assert isinstance(captured["registry"], ReadOnlyToolRegistry)


def test_delegate_policy_enforces_count_size_and_single_call():
    loop = AgentLoop.__new__(AgentLoop)
    loop.config = AgentLoopConfig(
        max_subagents_per_run=1,
        max_subagent_task_chars=20,
    )
    state = AgentState(task="inspect project")
    call = ToolCall("delegate_task", {"task": "inspect README"})
    state.current_tool_calls = [call]

    assert loop._check_tool_policy(state, call) is None

    state.current_tool_calls = [call, ToolCall("read_file", {})]
    assert "one task per" in loop._check_tool_policy(state, call)

    state.current_tool_calls = [call]
    assert loop._check_tool_policy(
        state,
        ToolCall("delegate_task", {"task": "x" * 21}),
    )
    state.metadata["delegated_tasks"] = ["inspect readme"]
    assert "already been attempted" in loop._check_tool_policy(state, call)


def test_delegate_execution_propagates_child_failure_and_metrics():
    child_state = AgentState(
        task="inspect README",
        completed=True,
        error="Maximum agent iterations reached.",
    )
    child_state.model_call_count = 2
    child_state.model_usage = {
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 28,
    }
    child_state.model_duration_ms = 35
    child_state.tool_executions.append(
        ToolExecution(
            tool_name="read_file",
            arguments={"path": "README.md"},
            duration_ms=12,
        )
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.registry = FakeRegistry()
    loop.config = AgentLoopConfig()
    loop.subagents = SimpleNamespace(run=lambda **_kwargs: child_state)
    state = AgentState(task="summarize the project")
    call = ToolCall("delegate_task", {"task": "inspect README"})

    execution = loop._execute_tool(state, call)

    assert execution.success is False
    assert "Maximum agent iterations" in execution.error
    assert state.unresolved_tool_failures["delegate_task"] == execution.error
    assert state.model_call_count == 2
    assert state.model_usage["total_tokens"] == 28
    assert state.model_duration_ms == 35
    assert state.metadata["subagent_tool_execution_count"] == 1
    assert state.metadata["subagent_tool_duration_ms"] == 12


def test_parent_exposes_and_owns_delegation_completion():
    class ParentModel:
        def __init__(self):
            self.planner_tools = []

        def chat(self, messages, tools=None, temperature=0.2):
            if tools is not None:
                self.planner_tools = tools
                return ModelResponse(
                    tool_calls=[
                        ToolCall(
                            "delegate_task",
                            {
                                "task": "inspect the README",
                                "reason_for_tool": "Independent documentation review",
                            },
                            "delegate-1",
                        )
                    ]
                )
            if messages[0].content.startswith(
                "You are the evidence and completion evaluator"
            ):
                return ModelResponse(
                    content=(
                        "EVIDENCE_SUFFICIENT: true\n"
                        "TASK_COMPLETE: true\n"
                        "GOAL_COMPLETE: true\n"
                        "REASON: Delegated findings are available.\n"
                        "NEXT_ACTION: FINISH"
                    )
                )
            return ModelResponse(content="The parent reviewed the findings.")

    class SuccessfulSubagent:
        calls = []

        def run(self, parent_task, delegated_task):
            self.calls.append((parent_task, delegated_task))
            result = AgentState(task=delegated_task)
            result.finish("README describes the local agent setup.")
            return result

    model = ParentModel()
    registry = FakeRegistry()
    agent = AgentLoop(
        model=model,
        mcp_client=object(),
        registry=registry,
        workspace=object(),
    )
    subagents = SuccessfulSubagent()
    agent.subagents = subagents

    result = agent.run("Summarize the project structure")

    assert result.completed
    assert result.error is None
    assert len(subagents.calls) == 1
    assert subagents.calls[0][1] == "inspect the README"
    assert model.planner_tools[-1]["function"]["name"] == "delegate_task"
    assert result.tool_executions[0].tool_name == "delegate_task"
    assert result.tool_executions[0].success
    assert result.final_answer == "The parent reviewed the findings."