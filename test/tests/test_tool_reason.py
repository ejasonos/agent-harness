import json

from mcp_client.registry import ToolRegistry
from mcp_client.schemas import MCPTool
from model.parser import ModelParser


def test_tool_schemas_require_a_reason_without_mutating_original_schema():
    original_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    registry = ToolRegistry(client=None)
    registry._tools["directory_tree"] = MCPTool(
        name="directory_tree",
        input_schema=original_schema,
    )

    for tool in (
        registry.to_ollama_tools()[0],
        registry.to_openai_tools()[0],
    ):
        schema = tool["function"]["parameters"]
        assert "reason_for_tool" in schema["properties"]
        assert "reason_for_tool" in schema["required"]

    assert "reason_for_tool" not in original_schema["properties"]


def test_parser_keeps_reason_out_of_tool_arguments():
    response = ModelParser.parse(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "directory_tree",
                                    "arguments": json.dumps(
                                        {
                                            "path": ".",
                                            "reason_for_tool": (
                                                "Inspect project structure."
                                            ),
                                        }
                                    ),
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )

    call = response.tool_calls[0]
    assert call.arguments == {"path": "."}
    assert call.reason_for_tool == "Inspect project structure."