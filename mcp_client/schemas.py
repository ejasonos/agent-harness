"""
Data structures used by the MCP client and tool registry.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolPolicy:
    """
    Describes the reasoning and knowledge requirements for using a tool.

    Prerequisites are semantic knowledge states rather than specific
    tool names. This keeps the policy system independent of the exact
    implementation used to acquire that knowledge.
    """

    prerequisites: list[str] = field(
        default_factory=list
    )

    verify_after: bool = False

    avoid_repeat_after_failure: bool = True


@dataclass
class MCPTool:
    """
    Normalized representation of an MCP tool.
    """

    name: str

    description: str = ""

    input_schema: dict[str, Any] = field(
        default_factory=dict
    )

    policy: ToolPolicy = field(
        default_factory=ToolPolicy
    )

    @property
    def model_schema(self) -> dict[str, Any]:
        """
        Return a schema suitable for presenting to the LLM.
        """

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class MCPToolResult:
    """
    Normalized MCP tool execution result.
    """

    tool_name: str

    success: bool

    content: Any = None

    structured_content: Any = None

    error: str | None = None

    raw: dict[str, Any] | None = None

    def as_text(self) -> str:
        """
        Convert the tool result into text suitable for model context.
        """

        if self.error:
            return f"[MCP ERROR] {self.error}"

        if self.structured_content is not None:
            return str(self.structured_content)

        if self.content is None:
            return ""

        if isinstance(self.content, str):
            return self.content

        if isinstance(self.content, list):
            parts: list[str] = []

            for item in self.content:
                if isinstance(item, dict):
                    text = item.get("text")

                    if text is not None:
                        parts.append(str(text))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))

            return "\n".join(parts)

        return str(self.content)