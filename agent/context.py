from __future__ import annotations

from agent.state import AgentState
from model.messages import Message


class ContextManager:
    """
    Builds the model context from the current agent state.

    The context contains both conversation history and a compact
    representation of the agent's acquired workspace knowledge.
    """

    def __init__(
        self,
        max_messages: int = 100,
    ) -> None:
        if max_messages < 1:
            raise ValueError(
                "max_messages must be at least 1"
            )

        self.max_messages = max_messages

    # -------------------------------------------------------------
    # CONTEXT BUILDING
    # -------------------------------------------------------------

    def build_messages(
        self,
        state: AgentState,
    ) -> list[Message]:
        """
        Return the messages that should be sent to the model.

        Workspace knowledge is injected as a system-style context
        message when relevant.
        """

        messages = list(state.messages)

        knowledge_message = self._build_knowledge_message(
            state
        )

        if knowledge_message is not None:
            messages = [
                knowledge_message,
                *messages,
            ]

        if len(messages) <= self.max_messages:
            return messages

        return self._trim_messages(
            messages
        )

    # -------------------------------------------------------------
    # KNOWLEDGE CONTEXT
    # -------------------------------------------------------------

    def _build_knowledge_message(
        self,
        state: AgentState,
    ) -> Message | None:
        """
        Build a compact description of what the agent currently knows
        about the workspace.
        """

        sections: list[str] = []

        if state.workspace_discovered:
            sections.append(
                "Workspace discovery: completed."
            )
        else:
            sections.append(
                "Workspace discovery: not completed."
            )

        if state.discovered_paths:
            paths = sorted(
                state.discovered_paths
            )

            sections.append(
                "Known paths:\n"
                + "\n".join(
                    f"- {path}"
                    for path in paths
                )
            )

        if state.inspected_files:
            files = sorted(
                state.inspected_files
            )

            sections.append(
                "Inspected files:\n"
                + "\n".join(
                    f"- {path}"
                    for path in files
                )
            )

        if state.failed_tool_calls:
            sections.append(
                "Some exact tool calls have already failed. "
                "Do not blindly repeat the same call."
            )

        if state.verification_pending:
            mutation = state.metadata.get(
                "verification_required_by",
                "the previous mutation",
            )
            sections.append(
                f"Verification is required after {mutation}. "
                "Do not finish or make another mutation until "
                "a verification tool succeeds."
            )

        if state.unresolved_tool_failures:
            failures = "\n".join(
                f"- {name}: {error}"
                for name, error in sorted(
                    state.unresolved_tool_failures.items()
                )
            )
            sections.append(
                "Unresolved tool failures; do not claim the "
                "requested operation succeeded:\n"
                + failures
            )

        if not sections:
            return None

        return Message(
            role="system",
            content=(
                "AGENT WORKSPACE KNOWLEDGE\n\n"
                + "\n\n".join(sections)
            ),
        )

    # -------------------------------------------------------------
    # MESSAGE TRIMMING
    # -------------------------------------------------------------

    def _trim_messages(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """
        Keep the beginning of the conversation and the most recent
        messages.
        """

        if self.max_messages == 1:
            return messages[-1:]

        first_message = messages[0]

        recent_count = self.max_messages - 1

        recent_messages = messages[
            -recent_count:
        ]

        return [
            first_message,
            *recent_messages,
        ]

    # -------------------------------------------------------------
    # MESSAGE HELPERS
    # -------------------------------------------------------------

    def add_user_message(
        self,
        state: AgentState,
        content: str,
    ) -> None:
        state.add_message(
            Message(
                role="user",
                content=content,
            )
        )

    def add_assistant_message(
        self,
        state: AgentState,
        content: str,
    ) -> None:
        state.add_message(
            Message(
                role="assistant",
                content=content,
            )
        )

    def add_tool_message(
        self,
        state: AgentState,
        content: str,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        state.add_message(
            Message(
                role="tool",
                content=f"tool_response: \n{content}",
                name=tool_name,
                tool_call_id=tool_call_id,
            )
        )