from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import requests

from .messages import Message, ModelResponse

from .parser import ModelParser
from utils.logging import log_event


class OllamaClient:
    """
    Client for communicating with a local Ollama server.

    The rest of the agent should interact with this class rather than
    calling Ollama directly.
    """

    def __init__(
        self,
        model: str = "gemma4:e4b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[Message | dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> ModelResponse:
        """
        Send a conversation to Ollama.

        Args:
            messages:
                Conversation messages.

            tools:
                Tool definitions exposed to the model.

            temperature:
                Sampling temperature.

        Returns:
            ModelResponse containing text and/or tool calls.
        """

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._normalize_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if tools:
            payload["tools"] = tools

        request_id = uuid.uuid4().hex
        endpoint = f"{self.base_url}/api/chat"
        started = time.perf_counter()
        log_event(
            "model.request.sent",
            provider="ollama",
            request_id=request_id,
            model=self.model,
            endpoint=endpoint,
            message_count=len(payload["messages"]),
            tool_count=len(tools or []),
        )
        log_event(
            "model.request.payload",
            level=logging.DEBUG,
            provider="ollama",
            request_id=request_id,
            payload=payload,
        )

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            log_event(
                "model.response.received",
                provider="ollama",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            log_event(
                "model.response.payload",
                level=logging.DEBUG,
                provider="ollama",
                request_id=request_id,
                payload=response.text,
            )
            response.raise_for_status()
            data = response.json()
            parsed_response = self._parse_response(data)
            log_event(
                "model.response.parsed",
                provider="ollama",
                request_id=request_id,
                content_length=len(parsed_response.content or ""),
                tool_calls=[call.tool_name for call in parsed_response.tool_calls],
                finish_reason=parsed_response.finish_reason,
            )
            log_event(
                "model.response.details",
                level=logging.DEBUG,
                provider="ollama",
                request_id=request_id,
                content=parsed_response.content,
                tool_calls=[call.to_dict() for call in parsed_response.tool_calls],
                raw=data,
            )
            return parsed_response
        except Exception as exc:
            log_event(
                "model.request.failed",
                level=logging.ERROR,
                provider="ollama",
                request_id=request_id,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                error=str(exc),
                exc_info=True,
            )
            raise

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

        return normalized

    def _parse_response(
            self,
            data: dict[str, Any],
            ) -> ModelResponse:
        return ModelParser.parse(data)

    def health_check(self) -> bool:
        """
        Check whether Ollama is reachable.
        """

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )

            return response.ok

        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """
        Return models currently available in Ollama.
        """

        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        return [
            model.get("name")
            for model in data.get("models", [])
            if model.get("name")
        ]