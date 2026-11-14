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
        usage = ModelParser._extract_usage(data)

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            raw=data,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _extract_usage(data: dict[str, Any]) -> dict[str, int | float]:
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            prompt_tokens = raw_usage.get(
                "prompt_tokens",
                raw_usage.get("input_tokens"),
            )
            completion_tokens = raw_usage.get(
                "completion_tokens",
                raw_usage.get("output_tokens"),
            )
        else:
            prompt_tokens = data.get("prompt_eval_count")
            completion_tokens = data.get("eval_count")

        usage: dict[str, int | float] = {}
        if isinstance(prompt_tokens, (int, float)) and not isinstance(prompt_tokens, bool):
            usage["prompt_tokens"] = prompt_tokens
        if isinstance(completion_tokens, (int, float)) and not isinstance(completion_tokens, bool):
            usage["completion_tokens"] = completion_tokens

        total_tokens = (
            raw_usage.get("total_tokens")
            if isinstance(raw_usage, dict)
            else None
        )
        if isinstance(total_tokens, (int, float)) and not isinstance(total_tokens, bool):
            usage["total_tokens"] = total_tokens
        elif "prompt_tokens" in usage or "completion_tokens" in usage:
            usage["total_tokens"] = (
                usage.get("prompt_tokens", 0)
                + usage.get("completion_tokens", 0)
            )
        return usage

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