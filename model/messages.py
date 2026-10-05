from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }

        if self.name:
            message["name"] = self.name

        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id

        '''
        if self.tool_calls:
            message["tool_calls"] = [
                call.to_ollama_dict()
                for call in self.tool_calls
            ]
        '''
        if self.tool_calls:
            message["tool_calls"] = [
                call.to_openai_dict()
                for call in self.tool_calls
            ]

        return message


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    reason_for_tool: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "reason_for_tool": self.reason_for_tool,
        }

    def to_ollama_dict(self) -> dict[str, Any]:
        arguments = dict(self.arguments)
        if self.reason_for_tool:
            arguments["reason_for_tool"] = self.reason_for_tool
        return {
            "function": {
                "name": self.tool_name,
                "arguments": arguments,
            }
        }

    def to_openai_dict(self) -> dict[str, Any]:
        import json

        arguments = dict(self.arguments)
        if self.reason_for_tool:
            arguments["reason_for_tool"] = self.reason_for_tool
        return {
            "id": self.call_id or "",
            "type": "function",
            "function": {
                "name": self.tool_name,
                "arguments": json.dumps(arguments)
            }
        }


@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    finish_reason: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final(self) -> bool:
        return not self.tool_calls