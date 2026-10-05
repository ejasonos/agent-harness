from __future__ import annotations

import json
from typing import Any

from .messages import ModelResponse, ToolCall


class ModelParser:
    """
    Normalizes model responses into the agent's internal format.

    Supports:
    - Ollama responses
    - OpenAI-compatible responses such as NVIDIA NIM
    """

    @staticmethod
    def parse(data: dict[str, Any]) -> ModelResponse:
        message = ModelParser._extract_message(data)

        content = message.get("content") or ""

        tool_calls = ModelParser.parse_tool_calls(
            message.get("tool_calls")
        )

        finish_reason = ModelParser._extract_finish_reason(data)

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            raw=data,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _extract_message(
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract the assistant message from either Ollama
        or OpenAI-compatible responses.
        """

        # Ollama
        message = data.get("message")

        if isinstance(message, dict):
            return message

        # OpenAI-compatible / NVIDIA
        choices = data.get("choices")

        if isinstance(choices, list) and choices:
            first_choice = choices[0]

            if isinstance(first_choice, dict):
                message = first_choice.get("message")

                if isinstance(message, dict):
                    return message

        return {}

    @staticmethod
    def _extract_finish_reason(
        data: dict[str, Any],
    ) -> str | None:

        # Ollama
        finish_reason = data.get("done_reason")

        if finish_reason:
            return finish_reason

        # OpenAI-compatible / NVIDIA
        choices = data.get("choices")

        if isinstance(choices, list) and choices:
            first_choice = choices[0]

            if isinstance(first_choice, dict):
                return first_choice.get("finish_reason")

        return None

    @staticmethod
    def parse_tool_calls(
        raw_tool_calls: Any,
    ) -> list[ToolCall]:

        if not raw_tool_calls:
            return []

        if not isinstance(raw_tool_calls, list):
            return []

        calls: list[ToolCall] = []

        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue

            function = raw_call.get("function")

            if not isinstance(function, dict):
                continue

            name = function.get("name")

            if not name:
                continue

            arguments = ModelParser.parse_arguments(
                function.get("arguments")
            )
            reason_for_tool = arguments.pop(
                "reason_for_tool",
                "",
            )
            if not isinstance(reason_for_tool, str):
                reason_for_tool = ""

            calls.append(
                ToolCall(
                    tool_name=name,
                    arguments=arguments,
                    call_id=raw_call.get("id"),
                    reason_for_tool=reason_for_tool,
                )
            )

        return calls

    @staticmethod
    def parse_arguments(
        arguments: Any,
    ) -> dict[str, Any]:

        if arguments is None:
            return {}

        if isinstance(arguments, dict):
            return arguments

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)

                if isinstance(parsed, dict):
                    return parsed

            except json.JSONDecodeError:
                pass

        return {}

    @staticmethod
    def extract_text(
        data: dict[str, Any],
    ) -> str:

        message = ModelParser._extract_message(data)

        content = message.get("content")

        if isinstance(content, str):
            return content

        return ""