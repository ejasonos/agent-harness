from types import SimpleNamespace
from unittest.mock import Mock

from agent.context import ContextManager
from agent.evaluator import NextAction
from agent.loop import AgentLoop
from agent.state import AgentState
from mcp_client.schemas import MCPToolResult
from model.messages import ModelResponse, ToolCall
from workspace.manager import WorkspaceManager


def test_application_failure_overrides_mcp_transport_success():
    loop = AgentLoop.__new__(AgentLoop)
    loop.registry = SimpleNamespace(get=lambda _name: object())
    loop.mcp_client = SimpleNamespace(
        call_tool=lambda *_args: MCPToolResult(
            tool_name="execute_command",
            success=True,
            structured_content={
                "result": {
                    "command": "git status",
                    "success": False,
                    "return_code": 128,
                    "stderr": "fatal: not a git repository",
                    "stdout": "",
                }
            },
        )
    )

    execution = loop._execute_tool(
        AgentState(task="run git status"),
        ToolCall(
            tool_name="execute_command",
            arguments={"command": "git status"},
        ),
    )

    assert execution.success is False
    assert "not a git repository" in execution.error


def test_agent_does_not_finalize_after_last_tool_failure():
    registry = SimpleNamespace(
        discover=lambda: [],
        to_openai_tools=lambda: [],
        get=lambda _name: object(),
    )
    loop = AgentLoop(
        model=object(),
        mcp_client=object(),
        registry=registry,
        workspace=WorkspaceManager("."),
    )
    failure = SimpleNamespace(
        tool_name="execute_command",
        success=False,
        error="fatal: not a git repository (exit code 128)",
        result=(
            '{"success": false, "return_code": 128, '
            '"stderr": "fatal: not a git repository"}'
        ),
    )

    def plan_with_failure(state, tools=None):
        state.add_tool_execution(failure)
        state.unresolved_tool_failures[
            "execute_command"
        ] = failure.error
        state.add_tool_execution(
            SimpleNamespace(
                tool_name="directory_tree",
                success=True,
                error=None,
                result="some unrelated successful output",
            )
        )
        return SimpleNamespace(
            response=ModelResponse(
                content="On branch main; working tree clean"
            )
        )

    loop.planner.plan = Mock(side_effect=plan_with_failure)
    loop.evaluator.evaluate = Mock(
        side_effect=AssertionError(
            "a failed command must not be treated as evidence"
        )
    )

    state = loop.run("Show me the output of git status")

    assert state.completed is True
    assert state.final_answer is None
    assert "not a git repository" in state.error
    loop.evaluator.evaluate.assert_not_called()


def test_evaluator_cannot_finish_with_unresolved_tool_failure():
    loop = AgentLoop.__new__(AgentLoop)
    loop.context = ContextManager()
    state = AgentState(task="run git status")
    state.unresolved_tool_failures["execute_command"] = (
        "fatal: not a git repository"
    )
    evaluation = SimpleNamespace(
        next_action=NextAction.FINISH,
        goal_complete=True,
        reason="On branch main; working tree clean",
    )

    assert loop._handle_evaluation(state, evaluation) is False
    assert state.completed is False