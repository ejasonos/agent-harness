from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from agent.context import ContextManager
from agent.evaluator import Evaluator, NextAction
from agent.planner import Planner
from agent.recovery import RecoveryManager
from agent.state import AgentState, ToolExecution
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.messages import Message, ToolCall
from model.ollama_client import OllamaClient
from workspace.manager import WorkspaceManager


VERIFICATION_TOOL_NAMES = {
    "read_file",
    "git_diff",
    "git_status",
    "run_tests",
    "run_test_file",
    "run_linter",
    "run_formatter_check",
    "run_typecheck",
    "run_build",
    "browser_screenshot",
    "image_info",
    "extract_document_text",
    "extract_pdf_text",
}


@dataclass
class AgentLoopConfig:
    max_iterations: int = 100
    max_tool_calls: int = 100


class AgentLoop:
    """
    Autonomous agent execution loop.

    The evaluator acts as a control boundary between execution phases.

    High-level flow:

        User task
            ↓
        Planner
            ↓
        Tool call?
          /       \
        Yes        No
         ↓          ↓
      Execute    Evaluate
         ↓          ↓
      Evaluate   ┌────────────────┐
         ↓       │ NextAction     │
         │       ├────────────────┤
         └──────→│ INVESTIGATE    │
                 │ SYNTHESIZE     │
                 │ MODIFY         │
                 │ VERIFY         │
                 │ FINISH         │
                 └────────────────┘
    """

    def __init__(
        self,
        model: OllamaClient,
        mcp_client: MCPClient,
        registry: ToolRegistry,
        workspace: WorkspaceManager,
        context: ContextManager | None = None,
        config: AgentLoopConfig | None = None,
    ) -> None:
        self.model = model
        self.mcp_client = mcp_client
        self.registry = registry
        self.workspace = workspace
        self.context = context or ContextManager()
        self.config = config or AgentLoopConfig()

        self.recovery = RecoveryManager(
            max_retries_per_tool=2
        )

        self.planner = Planner(
            model=self.model,
            context=self.context,
        )

        self.evaluator = Evaluator(
            model=self.model,
        )

    # =============================================================
    # MAIN LOOP
    # =============================================================

    def run(
        self,
        task: str,
    ) -> AgentState:
        state = AgentState(
            task=task
        )

        # ---------------------------------------------------------
        # INITIAL USER MESSAGE
        # ---------------------------------------------------------

        self.context.add_user_message(
            state,
            task,
        )

        # ---------------------------------------------------------
        # DISCOVER AVAILABLE TOOLS
        # ---------------------------------------------------------

        self.registry.discover()

        tools = self.registry.to_openai_tools()

        # ---------------------------------------------------------
        # MAIN AGENT LOOP
        # ---------------------------------------------------------

        for _ in range(
            self.config.max_iterations
        ):
            state.next_iteration()

            # -----------------------------------------------------
            # ASK PLANNER WHAT TO DO NEXT
            # -----------------------------------------------------

            plan = self.planner.plan(
                state,
                tools=tools,
            )

            response = plan.response

            # -----------------------------------------------------
            # MODEL DID NOT REQUEST A TOOL
            # -----------------------------------------------------

            if not response.has_tool_calls:
                if state.unresolved_tool_failures:
                    last_execution = next(
                        (
                            execution
                            for execution in reversed(
                                state.tool_executions
                            )
                            if not execution.success
                            and execution.tool_name
                            in state.unresolved_tool_failures
                        ),
                        None,
                    )

                    if last_execution is not None:
                        state.fail(
                            self._format_tool_failure(
                                last_execution
                            )
                        )
                    else:
                        failed_tool, error = next(
                            iter(
                                state.unresolved_tool_failures.items()
                            )
                        )
                        state.fail(
                            f"{failed_tool} failed. "
                            "The agent cannot report a successful result.\n"
                            f"{error}"
                        )
                    return state

                answer = (
                    response.content or ""
                ).strip()

                self.context.add_assistant_message(
                    state,
                    answer,
                )

                evaluation = self.evaluator.evaluate(
                    state=state,
                )

                self._store_evaluation(
                    state,
                    evaluation,
                )

                # -------------------------------------------------
                # HONOR EVALUATOR CONTROL DECISION
                # -------------------------------------------------

                if self._handle_evaluation(
                    state,
                    evaluation,
                    answer,
                ):
                    return state

                continue

            # -----------------------------------------------------
            # MODEL REQUESTED TOOLS
            # -----------------------------------------------------

            state.current_tool_calls = (
                response.tool_calls
            )

            state.add_message(
                Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            # -----------------------------------------------------
            # CHECK TOTAL TOOL-CALL LIMIT
            # -----------------------------------------------------

            total_calls = len(
                state.tool_executions
            )

            if (
                total_calls
                + len(response.tool_calls)
                > self.config.max_tool_calls
            ):
                state.fail(
                    "Maximum tool-call limit reached."
                )

                return state

            # -----------------------------------------------------
            # EXECUTE TOOL CALLS
            # -----------------------------------------------------

            stop_after_evaluation = False

            for tool_call in response.tool_calls:

                # -------------------------------------------------
                # CHECK TOOL POLICY
                # -------------------------------------------------

                policy_result = (
                    self._check_tool_policy(
                        state,
                        tool_call,
                    )
                )

                if policy_result is not None:
                    self.context.add_tool_message(
                        state,
                        content=policy_result,
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.call_id,
                    )

                    continue

                # -------------------------------------------------
                # EXECUTE TOOL
                # -------------------------------------------------

                execution = self._execute_tool(
                    state,
                    tool_call,
                )

                state.add_tool_execution(
                    execution
                )

                # -------------------------------------------------
                # TOOL FAILED
                # -------------------------------------------------

                if not execution.success:

                    state.record_failed_tool_call(
                        tool_call.tool_name,
                        tool_call.arguments,
                    )

                    recovery = (
                        self.recovery.evaluate(
                            tool_call.tool_name,
                            execution.error,
                            tool_call.arguments,
                        )
                    )

                    recovery_message = (
                        f"{recovery.message}\n"
                        f"tool_response: "
                        f"{execution.error}"
                    )

                    self.context.add_tool_message(
                        state,
                        content=recovery_message,
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.call_id,
                    )

                    if not recovery.should_retry:
                        state.fail(
                            recovery.message
                        )

                        return state

                    continue

                # -------------------------------------------------
                # TOOL SUCCEEDED
                # -------------------------------------------------

                self.recovery.reset(
                    tool_call.tool_name,
                    tool_call.arguments,
                )

                self.context.add_tool_message(
                    state,
                    content=str(
                        execution.result
                    ),
                    tool_name=tool_call.tool_name,
                    tool_call_id=tool_call.call_id,
                )

                # -------------------------------------------------
                # UPDATE AGENT KNOWLEDGE
                # -------------------------------------------------

                self._update_knowledge_state(
                    state,
                    tool_call,
                    execution,
                )

                registered_tool = self.registry.get(
                    tool_call.tool_name
                )

                if (
                    registered_tool is not None
                    and registered_tool.policy.verify_after
                ):
                    state.verification_pending = True
                    state.metadata[
                        "verification_required_by"
                    ] = tool_call.tool_name
                elif (
                    tool_call.tool_name
                    in VERIFICATION_TOOL_NAMES
                    and self._verification_result_succeeded(
                        execution
                    )
                ):
                    state.verification_pending = False
                    state.metadata.pop(
                        "verification_required_by",
                        None,
                    )

                if (
                    not state.unresolved_tool_failures
                    and not state.verification_pending
                    and self._is_direct_delete_request(
                        state.task,
                        execution,
                    )
                ):
                    deleted_path = execution.arguments["path"]
                    state.finish(
                        f"Deleted {deleted_path}."
                    )
                    return state

                # -------------------------------------------------
                # EVALUATE PROGRESS
                # -------------------------------------------------

                evaluation = (
                    self.evaluator.evaluate(
                        state=state,
                        tool_execution=execution,
                    )
                )

                self._store_evaluation(
                    state,
                    evaluation,
                )

                # -------------------------------------------------
                # HONOR EVALUATOR CONTROL DECISION
                # -------------------------------------------------

                if self._handle_evaluation(
                    state,
                    evaluation,
                ):
                    return state

                # -------------------------------------------------
                # STOP EXECUTING REMAINING TOOL CALLS IF THE
                # EVALUATOR HAS MOVED THE AGENT TO ANOTHER PHASE.
                # -------------------------------------------------

                if state.verification_pending or evaluation.next_action in {
                    NextAction.SYNTHESIZE,
                    NextAction.FINISH,
                }:
                    stop_after_evaluation = True
                    break

            # -----------------------------------------------------
            # CLEAR CURRENT TOOL CALLS
            # -----------------------------------------------------

            state.current_tool_calls = []

            if stop_after_evaluation:
                continue

        # ---------------------------------------------------------
        # MAXIMUM ITERATIONS REACHED
        # ---------------------------------------------------------

        state.fail(
            "Maximum agent iterations reached."
        )

        return state

    # =============================================================
    # EVALUATION CONTROL
    # =============================================================

    @staticmethod
    def _is_direct_delete_request(
        task: str,
        execution: ToolExecution,
    ) -> bool:
        if (
            execution.tool_name != "delete_file"
            or not execution.success
        ):
            return False

        path = execution.arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            return False

        task_words = re.findall(r"[a-z0-9]+", task.lower())
        target_words = re.findall(
            r"[a-z0-9]+",
            path.replace("\\", "/").split("/")[-1].lower(),
        )
        if not target_words or not any(
            task_words[index:index + len(target_words)] == target_words
            for index in range(
                len(task_words) - len(target_words) + 1
            )
        ):
            return False

        delete_verbs = {"delete", "remove", "erase", "unlink"}
        if sum(word in delete_verbs for word in task_words) != 1:
            return False

        if any(
            word in {
                "and",
                "also",
                "then",
                "after",
                "before",
                "while",
                "but",
                "not",
                "never",
                "create",
                "write",
                "edit",
                "modify",
                "update",
                "rename",
                "move",
                "copy",
                "verify",
                "check",
                "run",
                "test",
                "build",
                "inspect",
                "read",
                "search",
                "explain",
            }
            for word in task_words
        ):
            return False

        return True

    def _handle_evaluation(
        self,
        state: AgentState,
        evaluation,
        answer: str | None = None,
    ) -> bool:
        """
        Convert an evaluator decision into an execution boundary.

        Returns:

            True
                The agent run is finished.

            False
                The main loop should continue.
        """

        if state.unresolved_tool_failures:
            failures = "; ".join(
                f"{name}: {error}"
                for name, error in sorted(
                    state.unresolved_tool_failures.items()
                )
            )
            self.context.add_user_message(
                state,
                (
                    "A tool operation is still unresolved. "
                    "Do not report success or finish; retry it "
                    "with a corrected approach or report this "
                    f"failure accurately: {failures}"
                ),
            )
            return False

        if state.verification_pending:
            mutation = state.metadata.get(
                "verification_required_by",
                "the previous mutation",
            )
            self.context.add_user_message(
                state,
                (
                    f"Verification is still required after "
                    f"{mutation}. Use a suitable verification "
                    "tool and do not finish or make another "
                    "mutation until it succeeds."
                ),
            )
            return False

        # ---------------------------------------------------------
        # FINISH
        # ---------------------------------------------------------

        if evaluation.next_action == NextAction.FINISH:
            final_answer = (
                answer
                or evaluation.reason
            ).strip()

            state.finish(
                final_answer
            )

            return True

        # ---------------------------------------------------------
        # GOAL COMPLETE
        # ---------------------------------------------------------

        if evaluation.goal_complete:
            if answer:
                state.finish(
                    answer.strip()
                )
            else:
                state.finish(
                    evaluation.reason
                )

            return True

        # ---------------------------------------------------------
        # SYNTHESIZE
        # ---------------------------------------------------------

        if evaluation.next_action == NextAction.SYNTHESIZE:
            return self._synthesize_final_answer(
                state
            )

        # ---------------------------------------------------------
        # INVESTIGATE / MODIFY / VERIFY
        #
        # These remain active execution phases. The next planner
        # iteration will receive the evaluator decision through
        # runtime state and conversation context.
        # ---------------------------------------------------------

        self.context.add_user_message(
            state,
            (
                "The evaluator has assessed the current state "
                "of the task.\n\n"
                f"Evidence sufficient: "
                f"{evaluation.evidence_sufficient}\n"
                f"Task complete: "
                f"{evaluation.task_complete}\n"
                f"Goal complete: "
                f"{evaluation.goal_complete}\n\n"
                f"Evaluator reason:\n"
                f"{evaluation.reason}\n\n"
                f"Required next phase: "
                f"{evaluation.next_action.value}\n\n"
                "Continue working on the original user task. "
                "Respect the required next phase."
            ),
        )

        return False

    def _synthesize_final_answer(
        self,
        state: AgentState,
    ) -> bool:
        """
        Produce the final answer after the evaluator determines
        that sufficient evidence exists.

        This is deliberately separate from the planner/tool loop.
        The model must now synthesize from the evidence rather than
        continue investigating.
        """

        history = self.context.build_messages(state)
        conversation = "\n\n".join(
            f"[{message.role}] {message.content or ''}"
            for message in history
        )

        messages = [
            Message(
                role="system",
                content=(
                    "You are the final answer synthesizer for an "
                    "AI Agent that performs tasks using its available "
                    "tools.\n\n"
                    "The investigation phase is complete.\n"
                    "Do not call tools.\n"
                    "Do not request additional investigation.\n"
                    "Use the conversation and tool results supplied "
                    "by the user message as your evidence. Treat tool "
                    "results as the source of truth, preserve their "
                    "specific facts, and do not invent examples.\n\n"
                    "Answer the user's original task directly, "
                    "accurately, and concisely."
                ),
            ),
            Message(
                role="user",
                content=(
                    "Produce the final answer to the original "
                    "user task using the conversation and tool "
                    "results below.\n\n"
                    f"ORIGINAL TASK:\n{state.task}\n\n"
                    f"CONVERSATION AND TOOL RESULTS:\n{conversation}"
                ),
            ),
        ]

        try:
            response = self.model.chat(
                messages=messages,
            )

        except Exception as exc:
            state.fail(
                f"Final answer synthesis failed: {exc}"
            )

            return True

        answer = (
            response.content or ""
        ).strip()

        if not answer:
            state.fail(
                "Final answer synthesis returned an empty response."
            )

            return True

        self.context.add_assistant_message(
            state,
            answer,
        )

        state.finish(
            answer
        )

        return True

    # =============================================================
    # TOOL POLICY
    # =============================================================

    def _check_tool_policy(
        self,
        state: AgentState,
        tool_call: ToolCall,
    ) -> str | None:
        """
        Validate whether a tool call is allowed to execute.

        Returns:

            None
                Tool is allowed.

            str
                Tool cannot execute yet and the returned message
                is fed back into the model.
        """

        tool = self.registry.get(
            tool_call.tool_name
        )

        if tool is None:
            return (
                f"Unknown tool: "
                f"{tool_call.tool_name}"
            )

        if (
            state.verification_pending
            and tool_call.tool_name not in VERIFICATION_TOOL_NAMES
        ):
            return (
                "A previous mutation still requires verification. "
                "Call a verification tool before continuing."
            )

        policy = tool.policy

        # ---------------------------------------------------------
        # PREVENT EXACT FAILED CALL REPETITION
        # ---------------------------------------------------------

        if (
            policy.avoid_repeat_after_failure
            and state.has_failed_tool_call(
                tool_call.tool_name,
                tool_call.arguments,
            )
        ):
            return (
                "This exact tool call has already failed.\n\n"
                f"Tool: {tool_call.tool_name}\n"
                f"Arguments: {tool_call.arguments}\n\n"
                "Do not repeat the same call. "
                "Use the previous error to choose a different "
                "target, argument, or strategy."
            )

        # ---------------------------------------------------------
        # CHECK KNOWLEDGE PREREQUISITES
        # ---------------------------------------------------------

        target_path = self._knowledge_path(
            self._extract_path_argument(
                tool_call.arguments
            )
        )

        missing = [
            prerequisite
            for prerequisite in policy.prerequisites
            if (
                (
                    prerequisite in {
                        "target_discovered",
                        "target_inspected",
                    }
                    and target_path is None
                )
                or not state.has_knowledge(
                    prerequisite,
                    target_path,
                )
            )
            and not self._new_target_does_not_need_inspection(
                prerequisite,
                tool_call.arguments,
            )
        ]

        if not missing:
            return None

        # ---------------------------------------------------------
        # RESOLVE KNOWN PREREQUISITES
        # ---------------------------------------------------------

        unresolved = []

        for prerequisite in missing:

            resolved = (
                self._resolve_prerequisite(
                    state,
                    prerequisite,
                    target_path,
                )
            )

            if not resolved:
                unresolved.append(
                    prerequisite
                )

        if unresolved:
            return (
                "The requested tool cannot be executed yet "
                "because required agent knowledge is missing.\n\n"
                f"Tool: {tool_call.tool_name}\n"
                f"Missing prerequisites: "
                f"{', '.join(unresolved)}\n\n"
                "Acquire the required knowledge first, then "
                "retry the intended operation."
            )

        return None

    def _verification_result_succeeded(
        self,
        execution: ToolExecution,
    ) -> bool:
        if not execution.success:
            return False

        result: Any = execution.result

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                try:
                    result = ast.literal_eval(result)
                except (ValueError, SyntaxError):
                    return True

        return not self._contains_verification_failure(result)

    def _contains_verification_failure(
        self,
        result: Any,
    ) -> bool:
        if isinstance(result, dict):
            if result.get("success") is False:
                return True

            if result.get("timed_out") is True:
                return True

            return any(
                self._contains_verification_failure(value)
                for value in result.values()
                if isinstance(value, (dict, list, tuple))
            )

        if isinstance(result, (list, tuple)):
            return any(
                self._contains_verification_failure(value)
                for value in result
            )

        return False

    # =============================================================
    # PREREQUISITE RESOLUTION
    # =============================================================

    def _resolve_prerequisite(
        self,
        state: AgentState,
        prerequisite: str,
        target_path: str | None = None,
    ) -> bool:
        """
        Attempt to satisfy a semantic tool prerequisite.
        """

        if (
            prerequisite in {
                "target_discovered",
                "target_inspected",
            }
            and target_path is None
        ):
            return False

        if state.has_knowledge(
            prerequisite,
            target_path,
        ):
            return True

        if prerequisite == "workspace_discovered":
            return self._discover_workspace(
                state
            )

        if prerequisite == "target_discovered":
            return False

        if prerequisite == "target_inspected":
            return False

        return False

    def _discover_workspace(
        self,
        state: AgentState,
    ) -> bool:
        """
        Establish workspace knowledge through the registered
        directory discovery tool.
        """

        discovery_tool = (
            self._find_discovery_tool()
        )

        if discovery_tool is None:
            return False

        discovery_call = ToolCall(
            tool_name=discovery_tool.name,
            arguments={
                "path": "./",
                "max_depth": 4,
            },
            reason_for_tool=(
                "Discover workspace structure to satisfy tool prerequisites."
            ),
        )

        execution = self._execute_tool(
            state,
            discovery_call,
        )

        state.add_tool_execution(
            execution
        )

        if not execution.success:
            state.record_failed_tool_call(
                discovery_call.tool_name,
                discovery_call.arguments,
            )

            self.context.add_tool_message(
                state,
                content=str(
                    execution.error
                    or "Workspace discovery failed."
                ),
                tool_name=discovery_call.tool_name,
                tool_call_id=discovery_call.call_id,
            )

            return False

        self.context.add_tool_message(
            state,
            content=str(
                execution.result
            ),
            tool_name=discovery_call.tool_name,
            tool_call_id=discovery_call.call_id,
        )

        state.mark_workspace_discovered()

        self._extract_discovered_paths(
            state,
            execution.result,
        )

        return True

    def _find_discovery_tool(self):
        """
        Find the registered workspace discovery tool.
        """

        preferred_names = (
            "directory_tree",
            "workspace_tree",
            "project_tree",
            "list_directory",
            "list_files",
        )

        for name in preferred_names:
            tool = self.registry.get(
                name
            )

            if tool is not None:
                return tool

        return None

    # =============================================================
    # KNOWLEDGE UPDATES
    # =============================================================

    def _update_knowledge_state(
        self,
        state: AgentState,
        tool_call: ToolCall,
        execution: ToolExecution,
    ) -> None:
        """
        Update semantic agent knowledge after successful execution.
        """

        if not execution.success:
            return

        tool_name = (
            tool_call.tool_name.lower()
        )

        # ---------------------------------------------------------
        # WORKSPACE DISCOVERY
        # ---------------------------------------------------------

        if tool_name in {
            "directory_tree",
            "workspace_tree",
            "project_tree",
            "list_directory",
            "list_files",
        }:
            state.mark_workspace_discovered()

            self._extract_discovered_paths(
                state,
                execution.result,
            )

        # ---------------------------------------------------------
        # FILE INSPECTION
        # ---------------------------------------------------------

        if tool_name in {
            "read_file",
            "inspect_file",
        }:
            path = self._knowledge_path(
                self._extract_path_argument(
                    tool_call.arguments
                )
            )

            if path:
                state.mark_path_discovered(
                    path
                )

                state.mark_file_inspected(
                    path
                )

        # ---------------------------------------------------------
        # SEARCH
        # ---------------------------------------------------------

        if tool_name in {
            "search_files",
            "grep",
            "find_in_files",
        }:
            self._extract_discovered_paths(
                state,
                execution.result,
            )

    def _extract_path_argument(
        self,
        arguments: dict[str, Any],
    ) -> str | None:
        """
        Extract a likely file/path argument from a tool call.
        """

        for key in (
            "path",
            "file_path",
            "filepath",
            "file",
            "filename",
            "target",
            "source_path",
            "output_path",
        ):
            value = arguments.get(
                key
            )

            if isinstance(
                value,
                str,
            ) and value.strip():
                return value.strip()

        return None

    def _knowledge_path(
        self,
        path: str | None,
    ) -> str | None:
        if path is None:
            return None

        try:
            return str(
                self.workspace.resolve(path)
            )
        except (OSError, ValueError):
            return path

    def _new_target_does_not_need_inspection(
        self,
        prerequisite: str,
        arguments: dict[str, Any],
    ) -> bool:
        if prerequisite != "target_inspected":
            return False

        path = self._extract_path_argument(arguments)

        if path is None:
            return False

        try:
            return not self.workspace.resolve(path).exists()
        except (OSError, ValueError):
            return False

    def _extract_discovered_paths(
        self,
        state: AgentState,
        result: Any,
    ) -> None:
        """
        Extract obvious path-like lines from discovery/search output.
        """

        if result is None:
            return

        if isinstance(result, dict):
            for key in ("path", "file", "filename"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    state.mark_path_discovered(
                        self._knowledge_path(value.strip())
                        or value.strip()
                    )
            for value in result.values():
                if isinstance(value, (dict, list)):
                    self._extract_discovered_paths(
                        state,
                        value,
                    )
            return

        if isinstance(result, list):
            for value in result:
                self._extract_discovered_paths(
                    state,
                    value,
                )
            return

        text = str(result)
        tree_parts: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            cleaned = (
                line
                .replace("├──", "")
                .replace("└──", "")
                .replace("│", "")
                .replace("├─", "")
                .replace("└─", "")
                .replace("─", "")
                .strip()
            )

            if not cleaned:
                continue

            connector_index = max(
                raw_line.find("├──"),
                raw_line.find("└──"),
            )

            if connector_index >= 0:
                prefix = raw_line[:connector_index]
                depth = len(prefix) // 4
                name = raw_line[connector_index + 4:].strip()

                while len(tree_parts) > depth:
                    tree_parts.pop()

                tree_parts.append(name)
                discovered_path = self._knowledge_path(
                    "/".join(tree_parts)
                )
                if discovered_path:
                    state.mark_path_discovered(
                        discovered_path
                    )
                continue

            if cleaned.endswith("/"):
                discovered_path = self._knowledge_path(cleaned)
                if discovered_path:
                    state.mark_path_discovered(discovered_path)
                continue

            if (
                "." in cleaned
                or "/" in cleaned
                or "\\" in cleaned
            ):
                discovered_path = self._knowledge_path(cleaned)
                if discovered_path:
                    state.mark_path_discovered(discovered_path)

    # =============================================================
    # TOOL EXECUTION
    # =============================================================

    def _execute_tool(
        self,
        state: AgentState,
        tool_call: ToolCall,
    ) -> ToolExecution:
        execution = ToolExecution(
            tool_name=tool_call.tool_name,
            arguments=tool_call.arguments,
            reason_for_tool=tool_call.reason_for_tool,
        )

        # ---------------------------------------------------------
        # VERIFY TOOL EXISTS
        # ---------------------------------------------------------

        tool = self.registry.get(
            tool_call.tool_name
        )

        if tool is None:
            execution.success = False

            execution.error = (
                f"Unknown tool: "
                f"{tool_call.tool_name}"
            )
            state.unresolved_tool_failures[
                tool_call.tool_name
            ] = execution.error

            return execution

        # ---------------------------------------------------------
        # EXECUTE TOOL
        # ---------------------------------------------------------

        try:
            result = self.mcp_client.call_tool(
                tool_call.tool_name,
                tool_call.arguments,
            )

            payload = self._decode_tool_payload(result)
            execution.result = self._format_tool_payload(
                payload,
                result.as_text(),
            )

            payload_error = self._tool_payload_error(
                payload
            )

            execution.success = (
                result.success
                and payload_error is None
            )

            if not execution.success:
                execution.error = (
                    result.error
                    or payload_error
                    or "Tool execution failed."
                )
                state.unresolved_tool_failures[
                    tool_call.tool_name
                ] = execution.error
            else:
                state.unresolved_tool_failures.pop(
                    tool_call.tool_name,
                    None,
                )

        except Exception as exc:
            execution.success = False
            execution.error = str(
                exc
            )
            state.unresolved_tool_failures[
                tool_call.tool_name
            ] = execution.error

        return execution

    def _decode_tool_payload(
        self,
        result,
    ) -> Any:
        payload = result.structured_content

        if payload is None:
            payload = result.content

        if isinstance(payload, list):
            text_items = [
                item.get("text")
                for item in payload
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
            ]
            if len(text_items) == 1:
                payload = text_items[0]
            elif text_items:
                payload = "\n".join(text_items)

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                try:
                    payload = ast.literal_eval(payload)
                except (ValueError, SyntaxError):
                    return payload

        while (
            isinstance(payload, dict)
            and set(payload) == {"result"}
        ):
            payload = payload["result"]

        return payload

    def _format_tool_payload(
        self,
        payload: Any,
        fallback: str,
    ) -> str:
        if payload is None:
            return fallback

        if isinstance(payload, str):
            return payload

        try:
            return json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
        except (TypeError, ValueError):
            return str(payload)

    def _tool_payload_error(
        self,
        payload: Any,
    ) -> str | None:
        if not isinstance(payload, dict):
            return None

        if payload.get("success") is False:
            detail = (
                payload.get("stderr")
                or payload.get("error")
                or payload.get("message")
                or "Tool reported success=false."
            )
            return_code = payload.get("return_code")
            if return_code is not None:
                detail = f"{detail} (exit code {return_code})"
            return str(detail)

        if payload.get("timed_out") is True:
            return str(
                payload.get("error")
                or "Tool execution timed out."
            )

        return None

    def _format_tool_failure(
        self,
        execution: ToolExecution,
    ) -> str:
        details = execution.error or "No error details were returned."
        if execution.result:
            details = f"{details}\nTool result: {execution.result}"
        return (
            f"{execution.tool_name} failed. "
            "The agent cannot report a successful result.\n"
            f"{details}"
        )

    # =============================================================
    # EVALUATION
    # =============================================================

    def _store_evaluation(
        self,
        state: AgentState,
        evaluation,
    ) -> None:
        """
        Store the latest evaluator decision in runtime state.
        """

        state.metadata[
            "last_evaluation"
        ] = {
            "evidence_sufficient":
                evaluation.evidence_sufficient,
            "task_complete":
                evaluation.task_complete,
            "goal_complete":
                evaluation.goal_complete,
            "reason":
                evaluation.reason,
            "next_action":
                evaluation.next_action.value,
        }