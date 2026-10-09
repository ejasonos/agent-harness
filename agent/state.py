from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

from model.messages import Message, ToolCall


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

    # -------------------------------------------------------------
    # ITERATION STATE
    # -------------------------------------------------------------

    def next_iteration(self) -> None:
        self.iteration += 1

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