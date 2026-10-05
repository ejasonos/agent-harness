from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass
class RecoveryDecision:
    should_retry: bool
    message: str = ""


class RecoveryManager:
    """
    Manages recovery from failed tool executions.

    Retry state is tracked per specific tool invocation rather than
    only by tool name.

    This means:

        read_file("wrong.py")
            fails

    does not prevent:

        read_file("correct.py")

    from being attempted.
    """

    def __init__(
        self,
        max_retries_per_tool: int = 2,
    ) -> None:
        self.max_retries_per_tool = max_retries_per_tool

        self.retry_counts: dict[str, int] = {}

    # -------------------------------------------------------------
    # TOOL CALL IDENTITY
    # -------------------------------------------------------------

    def _call_key(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Create a stable identifier for a specific tool invocation.
        """

        try:
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

    # -------------------------------------------------------------
    # RECOVERY DECISION
    # -------------------------------------------------------------

    def evaluate(
        self,
        tool_name: str,
        error: str | None,
        arguments: dict[str, Any] | None = None,
    ) -> RecoveryDecision:
        """
        Determine whether a failed tool call should be retried.

        Retries are tracked per tool + arguments combination.
        """

        if not error:
            return RecoveryDecision(
                should_retry=False,
            )

        arguments = arguments or {}

        call_key = self._call_key(
            tool_name,
            arguments,
        )

        count = self.retry_counts.get(
            call_key,
            0,
        )

        if count >= self.max_retries_per_tool:
            return RecoveryDecision(
                should_retry=False,
                message=(
                    f"Maximum retries reached for "
                    f"{tool_name} with the same arguments."
                ),
            )

        next_attempt = count + 1

        self.retry_counts[call_key] = next_attempt

        return RecoveryDecision(
            should_retry=True,
            message=(
                f"Tool {tool_name} failed. "
                f"Attempt {next_attempt} of "
                f"{self.max_retries_per_tool} "
                f"for this specific call."
            ),
        )

    # -------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------

    def reset(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """
        Reset retry state for a specific successful tool call.

        If arguments are provided, only that exact invocation is reset.
        """

        arguments = arguments or {}

        call_key = self._call_key(
            tool_name,
            arguments,
        )

        self.retry_counts.pop(
            call_key,
            None,
        )