from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.context import ContextManager
from agent.state import AgentState
from model.messages import ModelResponse
from model.ollama_client import OllamaClient


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
        model: OllamaClient,
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

        messages = self.context.build_messages(state)

        response = self.model.chat(
            messages=messages,
            tools=tools,
        )

        return Plan(
            response=response,
        )