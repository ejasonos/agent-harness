from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.context import ContextManager
from agent.state import AgentState
from model.client import ModelClient
from model.messages import Message, ModelResponse
from utils.logging import log_context, log_event


@dataclass
class Plan:
    """
    Represents the model's decision for the next step.
    """

    response: ModelResponse

    @property
    def should_use_tools(self) -> bool:
        return self.response.has_tool_calls

    @property
    def should_finish(self) -> bool:
        return not self.response.has_tool_calls


class Planner:
    """
    Responsible for asking the model what the agent should do next.
    """

    def __init__(
        self,
        model: ModelClient,
        context: ContextManager | None = None,
    ) -> None:
        self.model = model
        self.context = context or ContextManager()

    def plan(
        self,
        state: AgentState,
        tools: list[dict[str, Any]] | None = None,
    ) -> Plan:
        """
        Ask the model to determine the next action.
        """

        planner_instructions = Message(
            role="system",
            content=(
                "You are the planner in a bounded ReAct loop: reason "
                "about the next useful step, act with an available tool "
                "when needed, then use its observation to reassess. "
                "Follow the evaluator's required phase and the runtime's "
                "tool policies. Do not claim completion because a tool "
                "succeeded; the evaluator controls completion and the "
                "runtime requires verification after mutations. For public "
                "web research, use web_search; for opening, interacting with, "
                "or inspecting a live page, use browser tools. Launch the "
                "browser first, wait for dynamic content when needed, inspect "
                "the page with browser_page_snapshot, and use "
                "browser_get_failed_requests or "
                "browser_get_console to diagnose failures. Do not "
                "use delegate_task except for an independent read-only "
                "investigation; the parent owns all changes and final "
                "completion. Do not reveal private chain-of-thought. "
                "Give only concise action rationales or the user-facing answer."
            ),
        )
        reserved_tokens = (
            self.context.estimate_tokens_for_value(tools or [])
            + self.context.token_counter(planner_instructions)
        )
        messages = self.context.build_messages(
            state,
            reserved_tokens=reserved_tokens,
        )
        messages.insert(0, planner_instructions)

        log_event(
            "agent.planner.requested",
            message_count=len(messages),
            tool_count=len(tools or []),
        )
        with log_context(model_purpose="planner"):
            response = self.model.chat(
                messages=messages,
                tools=tools,
            )
        log_event(
            "agent.planner.responded",
            content_length=len(response.content or ""),
            tool_calls=[call.tool_name for call in response.tool_calls],
        )

        return Plan(
            response=response,
        )