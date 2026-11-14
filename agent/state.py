from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

from model.messages import Message, ToolCall
from utils.task_journal import record_task_event


@dataclass
class ToolExecution:
    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    success: bool = False
    error: str | None = None
    duration_ms: float | None = None
    reason_for_tool: str = ""


@dataclass
class AgentState:
    """
    Runtime state for a single agent task.

    AgentState stores both execution history and the knowledge
    the agent has acquired about the workspace.

    Knowledge state is intentionally semantic rather than tied
    to specific tools. For example, the agent records that the
    workspace has been discovered, not merely that
    `directory_tree` was executed.
    """

    task: str

    messages: list[Message] = field(default_factory=list)

    tool_executions: list[ToolExecution] = field(
        default_factory=list
    )

    current_tool_calls: list[ToolCall] = field(
        default_factory=list
    )

    iteration: int = 0

    model_call_count: int = 0

    model_usage: dict[str, int | float] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )

    model_duration_ms: float = 0.0

    estimated_model_cost_usd: float | None = None

    run_duration_ms: float | None = None

    completed: bool = False

    verification_pending: bool = False

    final_answer: str | None = None

    error: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # -------------------------------------------------------------
    # WORKSPACE KNOWLEDGE
    # -------------------------------------------------------------

    workspace_discovered: bool = False

    discovered_paths: set[str] = field(
        default_factory=set
    )

    inspected_files: set[str] = field(
        default_factory=set
    )

    # -------------------------------------------------------------
    # FAILED TOOL CALLS
    # -------------------------------------------------------------

    failed_tool_calls: set[str] = field(
        default_factory=set
    )

    unresolved_tool_failures: dict[str, str] = field(
        default_factory=dict
    )

    tool_failure_counts: dict[str, int] = field(
        default_factory=dict
    )

    # -------------------------------------------------------------
    # MESSAGE STATE
    # -------------------------------------------------------------

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    # -------------------------------------------------------------
    # TOOL EXECUTION STATE
    # -------------------------------------------------------------

    def add_tool_execution(
        self,
        execution: ToolExecution,
    ) -> None:
        self.tool_executions.append(execution)
        record_task_event(
            "tool_result",
            tool_name=execution.tool_name,
            arguments=getattr(execution, "arguments", {}),
            success=execution.success,
            result=execution.result,
            error=execution.error,
            duration_ms=getattr(execution, "duration_ms", None),
            reason_for_tool=getattr(execution, "reason_for_tool", ""),
        )

    # -------------------------------------------------------------
    # ITERATION STATE
    # -------------------------------------------------------------

    def next_iteration(self) -> None:
        self.iteration += 1

    def record_model_call(
        self,
        usage: dict[str, int | float] | None = None,
        *,
        input_cost_per_million_tokens: float | None = None,
        output_cost_per_million_tokens: float | None = None,
    ) -> None:
        usage = usage or {}
        self.model_call_count += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.model_usage[key] += value
        duration_ms = usage.get("request_duration_ms", 0)
        if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
            self.model_duration_ms += duration_ms

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if (
            input_cost_per_million_tokens is not None
            and output_cost_per_million_tokens is not None
            and isinstance(prompt_tokens, (int, float))
            and not isinstance(prompt_tokens, bool)
            and isinstance(completion_tokens, (int, float))
            and not isinstance(completion_tokens, bool)
        ):
            if self.estimated_model_cost_usd is None:
                self.estimated_model_cost_usd = 0.0
            self.estimated_model_cost_usd += (
                prompt_tokens * input_cost_per_million_tokens
                + completion_tokens * output_cost_per_million_tokens
            ) / 1_000_000

    # -------------------------------------------------------------
    # COMPLETION STATE
    # -------------------------------------------------------------

    def finish(self, answer: str) -> None:
        self.completed = True
        self.final_answer = answer
        self.error = None

    def fail(self, error: str) -> None:
        self.completed = True
        self.error = error

    # -------------------------------------------------------------
    # KNOWLEDGE STATE
    # -------------------------------------------------------------

    def mark_workspace_discovered(
        self,
        paths: list[str] | None = None,
    ) -> None:
        """
        Record that the workspace structure has been discovered.
        """

        self.workspace_discovered = True

        if paths:
            for path in paths:
                self.mark_path_discovered(path)

    def mark_path_discovered(
        self,
        path: str,
    ) -> None:
        """
        Record that a specific path is known to exist.
        """

        self.discovered_paths.add(
            self._normalize_path(path)
        )

    def mark_file_inspected(
        self,
        path: str,
    ) -> None:
        """
        Record that a file has been inspected.
        """

        self.inspected_files.add(
            self._normalize_path(path)
        )

    @staticmethod
    def _normalize_path(path: str) -> str:
        return os.path.normcase(
            os.path.normpath(path)
        )

    def has_knowledge(
        self,
        prerequisite: str,
        target_path: str | None = None,
    ) -> bool:
        """
        Check whether a semantic prerequisite is satisfied.

        Supported prerequisites currently include:

            workspace_discovered
            target_discovered
            target_inspected
        """

        if prerequisite == "workspace_discovered":
            return self.workspace_discovered

        if prerequisite == "target_discovered":
            if target_path is not None:
                return (
                    self._normalize_path(target_path)
                    in self.discovered_paths
                )
            return bool(self.discovered_paths)

        if prerequisite == "target_inspected":
            if target_path is not None:
                return (
                    self._normalize_path(target_path)
                    in self.inspected_files
                )
            return bool(self.inspected_files)

        # Unknown prerequisites are considered unsatisfied.
        return False

    # -------------------------------------------------------------
    # FAILED CALL TRACKING
    # -------------------------------------------------------------

    def tool_call_key(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Create a stable identifier for a specific tool invocation.

        The arguments are normalized so that the same logical
        tool call can be recognized even if dictionary ordering
        changes.
        """

        try:
            import json

            serialized_arguments = json.dumps(
                arguments,
                sort_keys=True,
                default=str,
            )
        except Exception:
            serialized_arguments = str(arguments)

        return (
            f"{tool_name}:{serialized_arguments}"
        )

    def record_failed_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """
        Record a specific failed tool invocation.
        """

        self.failed_tool_calls.add(
            self.tool_call_key(
                tool_name,
                arguments,
            )
        )

    def has_failed_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        """
        Determine whether this exact tool invocation has already failed.
        """

        return (
            self.tool_call_key(
                tool_name,
                arguments,
            )
            in self.failed_tool_calls
        )