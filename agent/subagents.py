from __future__ import annotations

from typing import Any

from mcp_client.schemas import MCPTool


READ_ONLY_SUBAGENT_TOOLS = frozenset(
    {
        "directory_tree",
        "list_directory",
        "list_files",
        "workspace_tree",
        "project_tree",
        "detect_project",
        "project_info",
        "project_commands",
        "detect_framework",
        "git_status",
        "git_diff",
        "git_history",
        "read_file",
        "read_files",
        "inspect_file",
        "search_files",
        "grep",
        "find_in_files",
        "extract_text",
        "extract_document_text",
        "extract_pdf_text",
        "image_info",
        "html_to_text",
    }
)


class ReadOnlyToolRegistry:
    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def discover(self) -> list[MCPTool]:
        return [
            tool
            for tool in self.registry.all()
            if tool.name in READ_ONLY_SUBAGENT_TOOLS
        ]

    def all(self) -> list[MCPTool]:
        return self.discover()

    def get(self, name: str) -> MCPTool | None:
        if name not in READ_ONLY_SUBAGENT_TOOLS:
            return None
        return self.registry.get(name)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [
            schema
            for schema in self.registry.to_openai_tools()
            if schema.get("function", {}).get("name")
            in READ_ONLY_SUBAGENT_TOOLS
        ]


class SubagentRunner:
    def __init__(
        self,
        model: Any,
        mcp_client: Any,
        registry: Any,
        workspace: Any,
        config: Any,
    ) -> None:
        self.model = model
        self.mcp_client = mcp_client
        self.registry = registry
        self.workspace = workspace
        self.config = config

    def run(self, parent_task: str, delegated_task: str):
        from agent.loop import AgentLoop, AgentLoopConfig

        child_config = AgentLoopConfig(
            max_iterations=self.config.max_subagent_iterations,
            max_tool_calls=self.config.max_subagent_tool_calls,
            max_context_messages=self.config.max_context_messages,
            max_context_tokens=self.config.max_context_tokens,
            model_input_cost_per_million_tokens=(
                self.config.model_input_cost_per_million_tokens
            ),
            model_output_cost_per_million_tokens=(
                self.config.model_output_cost_per_million_tokens
            ),
            enable_subagents=False,
            max_subagents_per_run=0,
            max_subagent_iterations=self.config.max_subagent_iterations,
            max_subagent_tool_calls=self.config.max_subagent_tool_calls,
            max_subagent_task_chars=self.config.max_subagent_task_chars,
        )
        child = AgentLoop(
            model=self.model,
            mcp_client=self.mcp_client,
            registry=ReadOnlyToolRegistry(self.registry),
            workspace=self.workspace,
            config=child_config,
        )
        task = (
            f"Parent objective: {parent_task}\n\n"
            f"Delegated read-only investigation: {delegated_task}\n\n"
            "Return concise findings supported by tool observations. "
            "Do not modify files, run commands, or delegate further."
        )
        return child.run(task)