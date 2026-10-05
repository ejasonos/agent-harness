from __future__ import annotations

from typing import Any

import requests

from .messages import Message, ModelResponse
from .parser import ModelParser


class NVIDIAClient:
    """
    Client for NVIDIA's OpenAI-compatible inference API.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: int = 600,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not configured."
            )

    def chat(
        self,
        messages: list[Message | dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> ModelResponse:

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._normalize_messages(messages),
            "temperature": temperature,
            "stream": False,
        }

        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:
            print("\nNVIDIA API ERROR")
            print("Status:", response.status_code)
            print("Model:", self.model)
            print("URL:", f"{self.base_url}/chat/completions")
            print("Tool count:", len(tools or []))
            print("Response:", response.text)
            print("Payload:")
            print(payload)
            response.raise_for_status()

        data = response.json()

        return self._parse_response(data)

    def _normalize_messages(
        self,
        messages: list[Message | dict[str, Any]],
    ) -> list[dict[str, Any]]:

        normalized: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, Message):
                normalized.append(message.to_dict())

            elif isinstance(message, dict):
                normalized.append(message)

            else:
                raise TypeError(
                    f"Unsupported message type: {type(message).__name__}"
                )

        return self._split_multi_call_turns(normalized)

    @staticmethod
    def _split_multi_call_turns(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Serialize completed tool batches as single-call turns."""
        result: list[dict[str, Any]] = []
        index = 0

        while index < len(messages):
            message = messages[index]
            tool_calls = message.get("tool_calls")

            if (
                message.get("role") != "assistant"
                or not isinstance(tool_calls, list)
                or len(tool_calls) < 2
            ):
                result.append(message)
                index += 1
                continue

            call_ids = [
                call.get("id")
                for call in tool_calls
                if isinstance(call, dict)
            ]
            if (
                len(call_ids) != len(tool_calls)
                or any(not call_id for call_id in call_ids)
                or len(set(call_ids)) != len(call_ids)
            ):
                result.append(message)
                index += 1
                continue

            responses: dict[str, dict[str, Any]] = {}
            cursor = index + 1
            while cursor < len(messages):
                tool_message = messages[cursor]
                if tool_message.get("role") != "tool":
                    break
                tool_call_id = tool_message.get("tool_call_id")
                if tool_call_id not in call_ids:
                    break
                responses[tool_call_id] = tool_message
                cursor += 1

            if any(call_id not in responses for call_id in call_ids):
                result.append(message)
                index += 1
                continue

            for tool_call, call_id in zip(tool_calls, call_ids):
                result.append(
                    {
                        **message,
                        "tool_calls": [tool_call],
                    }
                )
                result.append(responses[call_id])

            index = cursor

        return result

    def _parse_response(
        self,
        data: dict[str, Any],
    ) -> ModelResponse:

        return ModelParser.parse(data)