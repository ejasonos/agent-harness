from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from agent.context import ContextManager
from agent.evaluator import NextAction
from agent.loop import AgentLoop
from agent.state import AgentState, ToolExecution
from mcp_client.registry import ToolRegistry
from mcp_client.schemas import MCPTool, ToolPolicy
from model.messages import ToolCall
from workspace.manager import WorkspaceManager


def test_policy_allows_call_after_prerequisite_resolution():
    tool = SimpleNamespace(
        policy=SimpleNamespace(
            avoid_repeat_after_failure=False,
            prerequisites=["workspace_discovered"],
        )
    )
    state = AgentState(task="inspect the project")
    loop = AgentLoop.__new__(AgentLoop)
    loop.registry = SimpleNamespace(get=lambda _name: tool)
    loop.workspace = WorkspaceManager(".")
    loop._resolve_prerequisite = Mock(return_value=True)

    result = loop._check_tool_policy(
        state,
        ToolCall(
            tool_name="read_file",
            arguments={"path": "README.md"},
        ),
    )

    assert result is None
    resolved_prerequisite = (
        loop._resolve_prerequisite.call_args.args
    )
    assert resolved_prerequisite[:2] == (
        state,
        "workspace_discovered",
    )


def test_target_inspection_applies_to_the_requested_path():
    with TemporaryDirectory() as directory:
        workspace = WorkspaceManager(directory)
        target = workspace.root / "src" / "target.py"
        target.parent.mkdir()
        target.write_text("print('target')", encoding="utf-8")

        tool = MCPTool(
            name="edit_file",
            policy=ToolPolicy(
                prerequisites=[
                    "workspace_discovered",
                    "target_inspected",
                ]
            ),
        )
        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop.registry = SimpleNamespace(get=lambda _name: tool)
        state = AgentState(task="edit target")
        state.workspace_discovered = True
        state.mark_file_inspected("other.py")
        call = ToolCall(
            tool_name="edit_file",
            arguments={"path": "src/target.py"},
        )

        blocked = loop._check_tool_policy(state, call)
        assert blocked is not None
        assert "target_inspected" in blocked

        state.mark_file_inspected(str(target))
        assert loop._check_tool_policy(state, call) is None

        pathless_call = ToolCall(
            tool_name="edit_file",
            arguments={},
        )
        assert loop._check_tool_policy(
            state,
            pathless_call,
        ) is not None


def test_new_file_does_not_require_prior_inspection():
    with TemporaryDirectory() as directory:
        workspace = WorkspaceManager(directory)
        tool = MCPTool(
            name="write_file",
            policy=ToolPolicy(
                prerequisites=[
                    "workspace_discovered",
                    "target_inspected",
                ]
            ),
        )
        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop.registry = SimpleNamespace(get=lambda _name: tool)
        state = AgentState(task="create a file")
        state.workspace_discovered = True

        assert loop._check_tool_policy(
            state,
            ToolCall(
                tool_name="write_file",
                arguments={"path": "new.py"},
            ),
        ) is None


def test_directory_tree_records_nested_paths():
    with TemporaryDirectory() as directory:
        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = WorkspaceManager(directory)
        state = AgentState(task="discover files")

        loop._extract_discovered_paths(
            state,
            "root\n├── src\n│   └── target.py",
        )

        target = str(loop.workspace.resolve("src/target.py"))
        assert state.has_knowledge(
            "target_discovered",
            target,
        )


def test_registered_tool_names_receive_policies():
    registry = ToolRegistry(None)

    assert registry._default_policy_for_tool(
        "browser_goto"
    ).prerequisites == []
    assert registry._default_policy_for_tool(
        "browser_page_text"
    ).prerequisites == []
    assert registry._default_policy_for_tool(
        "web_download"
    ).prerequisites == [
        "workspace_discovered",
        "target_inspected",
    ]
    assert not registry._default_policy_for_tool(
        "delete_file"
    ).verify_after


def test_direct_successful_delete_request_can_finish_immediately():
    execution = ToolExecution(
        tool_name="delete_file",
        arguments={"path": "package.json"},
        success=True,
    )

    assert AgentLoop._is_direct_delete_request(
        "Delete the package.json file.",
        execution,
    )
    assert not AgentLoop._is_direct_delete_request(
        "Delete package.json and update the lockfile.",
        execution,
    )
    assert not AgentLoop._is_direct_delete_request(
        "Delete package.json.",
        ToolExecution(
            tool_name="delete_file",
            arguments={"path": "other.json"},
            success=True,
        ),
    )


def test_pending_verification_blocks_completion_and_other_tools():
    loop = AgentLoop.__new__(AgentLoop)
    loop.context = ContextManager()
    loop.registry = SimpleNamespace(
        get=lambda name: MCPTool(name=name, policy=ToolPolicy())
    )
    state = AgentState(
        task="edit a file",
        verification_pending=True,
    )

    evaluation = SimpleNamespace(
        next_action=NextAction.FINISH,
        goal_complete=True,
        reason="done",
    )
    assert loop._handle_evaluation(state, evaluation) is False
    assert state.completed is False
    assert loop._check_tool_policy(
        state,
        ToolCall(tool_name="edit_file", arguments={}),
    ) is not None
    assert loop._check_tool_policy(
        state,
        ToolCall(tool_name="run_tests", arguments={}),
    ) is None


def test_failed_verification_result_keeps_verification_pending():
    loop = AgentLoop.__new__(AgentLoop)

    assert not loop._verification_result_succeeded(
        ToolExecution(
            tool_name="run_tests",
            arguments={},
            result="{'success': False}",
            success=True,
        )
    )
    assert loop._verification_result_succeeded(
        ToolExecution(
            tool_name="run_tests",
            arguments={},
            result="{'success': True}",
            success=True,
        )
    )


def test_final_synthesis_receives_tool_results():
    loop = AgentLoop.__new__(AgentLoop)
    loop.context = ContextManager()
    loop.model = Mock(
        return_value=SimpleNamespace(
            content="The requirements include fastmcp and requests."
        )
    )
    state = AgentState(
        task="Explain requirements.txt",
    )
    loop.context.add_user_message(
        state,
        state.task,
    )
    loop.context.add_tool_message(
        state,
        "fastmcp\nrequests\npytest",
        tool_name="read_file",
    )

    assert loop._synthesize_final_answer(state)

    synthesis_prompt = loop.model.chat.call_args.kwargs[
        "messages"
    ][-1].content
    assert state.task in synthesis_prompt
    assert "fastmcp\nrequests\npytest" in synthesis_prompt