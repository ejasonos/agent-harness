from __future__ import annotations

from typing import Any, Protocol

from .messages import Message, ModelResponse


class ModelClient(Protocol):
    def chat(
        self,
        messages: list[Message | dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> ModelResponse:
        ...
